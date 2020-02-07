# %% Imports
from __future__ import unicode_literals, print_function

import itertools
import math
import random

from allennlp.data import Token
from allennlp.models import Model
import torch
import torch.nn as nn
from torch.autograd import Variable

import constants
from config import FLAGS
from constants import PADDING_TOKEN, MASKING_TOKEN, TO_BE_DELETED_TOKEN, EOS_TOKEN


class FullModel(Model):
    def __init__(self,vocab):
        """
        """
        super().__init__(vocab)
        self.vocab = vocab
        self.embedder = AlbertEmbedder(vocab)
        self.encoder = nn.Sequential(*[EncoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.decoder = MySequential(*[DecoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.predictor = nn.Linear(FLAGS.d_hidden, self.vocab.get_vocab_size()) #TODO maybe make a factorized ALBERT-like de-embedder as well? also share weights
        self.loss = nn.CrossEntropyLoss()
        


    def process_targets_for_loss(self, target_tokens,max_target_seq_length):
        padding_index = self.vocab.get_token_index(PADDING_TOKEN)
        current_target_seq_length = target_tokens.shape[1]
        padder = nn.ConstantPad1d((0, max_target_seq_length - current_target_seq_length), padding_index)
        target_tokens_contiguous = padder(target_tokens).contiguous().view(-1)

        return target_tokens_contiguous

    def forward(self, inputs, targets=None):
        input_tokens, target_tokens = inputs['tokens'], targets['tokens']
        d_batch = input_tokens.shape[0] # Actual batch size (might not equal FLAGS.d_batch, eg when not enough samples to fill the last batch
        max_target_seq_length = int(FLAGS.max_seq_length*FLAGS.masking_fraction*2 + 1) # Longest length if no adjacent masks
        targets = self.process_targets_for_loss(target_tokens,max_target_seq_length)

        embedded_inputs = self.embedder(input_tokens)
        encoded = self.encoder(nn.Dropout(p=FLAGS.dropout_rate)(embedded_inputs))

        output_tokens = [constants.DECODER_START_TOKEN] + [constants.PADDING_TOKEN for _ in range(max_target_seq_length - 1)]
        embedded_outputs = self.embedder(torch.tensor([[self.vocab.get_token_index(token) for token in output_tokens] for _ in range(d_batch)]))

        # Creating decoded sequence element by element, each time attending to preceding outputs from the decoder stack
        decoded = torch.zeros(d_batch, max_target_seq_length,FLAGS.d_hidden)
        for i in range(max_target_seq_length):
            embedded_outputs,_ = self.decoder(nn.Dropout()(embedded_outputs),encoded) #TODO figure out whether to, in each loop, give decoder output of decoded so far, and zeros else, as input, or pad elsewhere, ..
            decoded[:,i] = decoded_element[:,i] #TODO maybe should do teacher forcing: give embedding of actual word: correct word at training, predicted word at inference time
        prediction_distribution = nn.Softmax(dim=-1)(self.predictor(nn.Dropout(p=FLAGS.dropout_rate)(decoded)))
        prediction_distribution_contiguous = prediction_distribution.contiguous().view(-1,self.vocab.get_vocab_size())
        loss =  self.loss(prediction_distribution_contiguous,targets)
        return {'loss':loss, 'prediction_distribution':prediction_distribution} # Dictionary format for AllenNLP trainer loop

    def decode(self, output_dict):
        '''
        Overrides the AllenNLP decode method.
        Returns an actual best guess, needed at inference time, based on the probability distribution produced by :forward:
        '''
        prediction = output_dict['prediction_distribution'].max(-1)[1]
        return {'prediction':prediction}


def layer_normalize(param):
    mean = torch.mean(param, -1)
    mean_expanded = mean.unsqueeze(-1).expand(*(mean.shape + (param.shape[-1],)))
    st_dev = torch.sqrt(torch.mean((param - mean_expanded)**2, -1))
    st_dev_expanded = st_dev.unsqueeze(-1).expand(*(mean.shape + (param.shape[-1],)))
    return (param - mean_expanded) / st_dev_expanded

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiHeadAttention() #TODO add padding mask
        # Dropout after every feedforward layer
        self.feedforward = nn.Sequential(*[layer for _ in range(FLAGS.nb_feedforward_layers) for layer in (nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden),nn.Dropout(p=FLAGS.dropout_rate)) ]) #TODO  The feed-forward networks in each block consist of a dense layer with an output dimensionality of dff = 3072 followed by a ReLU nonlinearity and another dense layer

    def forward(self, input):
        att_out = layer_normalize(self.multihead_attention(input,input) + nn.Dropout(p=FLAGS.dropout_rate)(input))  # Include skip-connection
        ff_out = layer_normalize(self.feedforward(att_out) + nn.Dropout(p=FLAGS.dropout_rate)(att_out))
        return ff_out

class MySequential(nn.Sequential):
    '''
    Allows Sequential to pass on multiple in- and outputs
    '''
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention(use_causal_mask=True)
        self.attention =  MultiHeadAttention()
        # Dropout after every feedforward layer
        self.feedforward = nn.Sequential(*[layer for _ in range(FLAGS.nb_feedforward_layers) for layer in (nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden),nn.Dropout(p=FLAGS.dropout_rate)) ])


    def forward(self, output, original_encoded_input):
        output = layer_normalize(output)
        self_att_out = layer_normalize(self.self_attention(query=output, values=output) + nn.Dropout(p=FLAGS.dropout_rate)(output))  # Include skip-connection and layer normalization
        att_out = layer_normalize(self.attention(query=self_att_out, values=original_encoded_input ))
        ff_out = layer_normalize(self.feedforward(att_out) + nn.Dropout(p=FLAGS.dropout_rate)(att_out))
        return ff_out, original_encoded_input

class MultiHeadAttention(nn.Module):
    # relative position embedding with d_emb=1 (aka a scalar), shared across layers, but different between attention heads, in line with t5 paper
    # nb of possible relative positions ranges from - max_seq_length to + max_seq_length
    position_embedding = Variable(torch.rand(FLAGS.nb_heads,FLAGS.max_seq_length*2).cuda(FLAGS.device_idx),requires_grad=True)

    def __init__(self,use_causal_mask=False):
        super().__init__()
        self.project_q = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_k = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_v = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_o = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden) # In line with og t5 code (although not obvious from paper): there for if different d_head_hidden than (FLAGS.d_hidden // FLAGS.nb_heads), here that's not supported atm
        self.use_causal_mask = use_causal_mask
        self.relative_attention_bias = nn.Embedding(FLAGS.relative_attention_num_buckets, FLAGS.nb_heads)

    def forward(self, query,values,padding_mask): # Padding mask to make sure we don't attend to padded positions
        d_batch = query.shape[0]
        q = self.project_q(query)
        k = self.project_k(values)
        v = self.project_v(values)
        assert FLAGS.d_hidden % FLAGS.nb_heads == 0
        d_head_hidden = FLAGS.d_hidden // FLAGS.nb_heads
        query_length = query.shape[1]
        value_length = values.shape[1]

        # This reshaping slices the last dimension and stacks those slices along the first dimension
        # In the resulting first dimension, first come all slices from the first batch, then from the second, and so on
        # This is relevant for how to add the position embeddings: they are the same per batch, but not per slice
        q_multi_parts = q.contiguous().view(d_batch * FLAGS.nb_heads, query_length, d_head_hidden)
        k_multi_parts = k.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)
        v_multi_parts = v.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)

        att_weights = torch.bmm(q_multi_parts, k_multi_parts.transpose(1, 2)) # shape [d_batch x nb_heads, query_length, value_length]

        if self.use_causal_mask:
            causal_mask = torch.tensor([[[1 if value_index <= query_index else 0 for value_index in range(att_weights.shape[2])] for query_index in range(att_weights.shape[1])] for _ in range(att_weights.shape[0])]).cuda(FLAGS.device_idx)
            att_weights *= causal_mask


        att_weights = nn.Softmax(dim=-1)(att_weights)
        att_weights = nn.Dropout(p=FLAGS.dropout_rate)(att_weights)

        batch_pos_embeddings = self.select_pos_embeddings(query_length, value_length, d_batch)
        att_output_multi_parts = torch.bmm(att_weights + batch_pos_embeddings, v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(d_batch, query_length, FLAGS.d_hidden)
        att_output = self.project_o(att_output)
        return att_output

    def select_pos_embeddings(self, query_length, value_length, d_batch):
        rel_pos_indices = torch.tensor(
            [[q_idx - k_idx for k_idx in range(value_length)] for q_idx in range(query_length)])\
            .cuda(FLAGS.device_idx) # shape [nb_heads, query_length, value_length]
        bucket_idxs = MultiHeadAttention._relative_position_bucket(rel_pos_indices)
        single_pos_embeddings = self.relative_attention_bias(bucket_idxs)
        batch_pos_embeddings = single_pos_embeddings.repeat(1,1,d_batch).permute(2,0,1)
        return batch_pos_embeddings

    # Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_t5.py
    # Allow for position embeddings to be able to deal with longer distances
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        bucket_indices = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            # This is so buckets for negative numbers start after the buckets for positive ones
            bucket_indices += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        # This assigns bucket from larger numbers with an offset of max_exact, and then assigns numbers up to max_distance to the range max_exact - num buckets logarithmically
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        #This puts all values larger than max_distance in the bucket for biggest numbers
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        bucket_indices += torch.where(is_small, n, val_if_large)
        return bucket_indices


class AlbertEmbedder(nn.Module):
    """
    Factorized embedder, as proposed in the ALBERT paper: http://arxiv.org/abs/1909.11942
    """
    def __init__(self,vocab):
        super().__init__()
        self.idx_to_embedding = nn.Embedding(vocab.get_vocab_size('tokens'), FLAGS.d_emb)
        self.embedding_to_hidden = nn.Linear(FLAGS.d_emb, FLAGS.d_hidden)
    def forward(self, idxs):
        embedded = self.idx_to_embedding(idxs)
        hidden = self.embedding_to_hidden(embedded)
        return hidden


def t5_denoise_spans_objective(tokens): # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683
    '''
    Produces inputs and targets.
    Inputs correspond to the original tokens, with a certain fraction of tokens replaced by a MASK-token.
    Contiguous tokens that happen to get masked get replaced by a single MASK token.
    There is no switching with random words, ... ( “MASS-style” objective )
    Targets look like: [mask_0, *<first word that was masked, possibly multiple if contiguous>, mask_1, <same>, ... , mask_eos, padding, padding, ... >
    '''
    masked_indices = sorted(random.sample(range(len(tokens)), int(len(tokens) * FLAGS.masking_fraction)))  #

    given = [t if (i not in masked_indices) else MASKING_TOKEN for i, t in enumerate(tokens)]
    masked_given = [i for i, j in zip(given[1:], given[:-1]) if not (i == MASKING_TOKEN and i == j)]
    mask_counter = itertools.count()
    unique_masked_given = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_given]

    target = [tokens[i] for i in masked_indices]
    include_mask = [True] + [((i - j) != 1) for i, j in zip(masked_indices[1:], masked_indices[:-1])]
    masks = [MASKING_TOKEN if x else TO_BE_DELETED_TOKEN for x in include_mask]
    masked_target = [i for j in zip(masks, target) for i in j if i != TO_BE_DELETED_TOKEN]
    mask_counter = itertools.count()  # Restart count
    unique_masked_target = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_target] + [
        Token(EOS_TOKEN)]
    return unique_masked_given, unique_masked_target