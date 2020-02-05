# %% Imports
from __future__ import unicode_literals, print_function

import itertools
import random

from allennlp.data import Token
from allennlp.models import Model
import torch
import torch.nn as nn
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
        self.predictor = nn.Linear(FLAGS.d_hidden, self.vocab.get_vocab_size()) #TODO maybe make a factorized ALBERT-like de-embedder as well?
        self.loss = nn.CrossEntropyLoss()
        


    def process_targets_for_loss(self, target_tokens,max_target_seq_length):
        padding_index = self.vocab.get_token_index(PADDING_TOKEN)
        current_target_seq_length = target_tokens.shape[1]
        padder = nn.ConstantPad1d((0, max_target_seq_length - current_target_seq_length), padding_index)
        target_tokens_contiguous = padder(target_tokens).contiguous().view(-1)

        return target_tokens_contiguous

    def forward(self, inputs, targets=None):
        input_tokens, target_tokens = inputs['tokens'], targets['tokens']
        # if len(input_tokens.shape) == 1: # Dealing with non-batched input in total way #TODO delete this
        #     input_tokens = input_tokens.unsqueeze(0)
        #     target_tokens = target_tokens.unsqueeze(0)
        d_batch = input_tokens.shape[0] # Actual batch size (might not equal FLAGS.d_batch, eg when not enough samples to fill the last batch
        max_target_seq_length = int(FLAGS.max_seq_length*FLAGS.masking_fraction*2 + 1) # Longest length if no adjacent masks
        targets = self.process_targets_for_loss(target_tokens,max_target_seq_length)
        embedded_inputs = self.embedder(input_tokens)
        embedded_outputs = torch.rand(d_batch, max_target_seq_length,FLAGS.d_hidden).cuda(FLAGS.device_idx)
        encoded = self.encoder(nn.Dropout()(embedded_inputs))
        decoded,_ = self.decoder(nn.Dropout()(embedded_outputs),encoded)
        prediction_distribution = nn.Softmax(dim=-1)(self.predictor(nn.Dropout()(decoded)))
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
        self.multihead_attention = MultiHeadAttention()
        # Dropout after every feedforward layer
        self.feedforward = nn.Sequential(*[layer for _ in range(FLAGS.nb_feedforward_layers) for layer in (nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden),nn.Dropout()) ])

    def forward(self, input):
        att_out = layer_normalize(self.multihead_attention(input,input) + nn.Dropout()(input))  # Include skip-connection
        ff_out = layer_normalize(self.feedforward(att_out) + nn.Dropout()(att_out))
        return ff_out

class MySequential(nn.Sequential):
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
        self.feedforward = nn.Sequential(*[layer for _ in range(FLAGS.nb_feedforward_layers) for layer in (nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden),nn.Dropout()) ])


    def forward(self, output, original_encoded_input):
        output = layer_normalize(output)
        self_att_out = layer_normalize(self.self_attention(query=output, values=output) + nn.Dropout()(output))  # Include skip-connection and layer normalization
        att_out = layer_normalize(self.attention(query=self_att_out, values=original_encoded_input ))
        ff_out = layer_normalize(self.feedforward(att_out) + nn.Dropout()(att_out))
        return ff_out, original_encoded_input

class MultiHeadAttention(nn.Module):
    # relative position embedding with d_emb=1 (aka a scalar), shared across layers, but different between attention heads, in line with t5 paper
    # nb of possible relative positions ranges from - max_seq_length to + max_seq_length
    position_embedding = torch.rand(FLAGS.nb_heads,FLAGS.max_seq_length*2).cuda(FLAGS.device_idx)

    def __init__(self,use_causal_mask=False):
        super().__init__()
        self.project_q = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_k = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_v = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.use_causal_mask = use_causal_mask

    def forward(self, query,values):
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

        att_weights = nn.Dropout()(att_weights)
        batch_pos_embeddings = self.select_pos_embeddings(query_length, value_length, d_batch)
        att_output_multi_parts = torch.bmm(att_weights + batch_pos_embeddings, v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(d_batch, query_length, FLAGS.d_hidden)
        return att_output

    def select_pos_embeddings(self, query_length, value_length, d_batch):
        rel_pos_indices = torch.tensor(
            [[[q_idx - k_idx for k_idx in range(value_length)] for q_idx in range(query_length)] for _ in
             range(FLAGS.nb_heads)]).cuda(FLAGS.device_idx) # shape [nb_heads, query_length, value_length]
        rel_pos_indices += FLAGS.max_seq_length  # Because torch.gather doesn't work with negative indices
        single_pos_embeddings = torch.cat(
            [MultiHeadAttention.position_embedding.gather(1, rel_pos_indices[:, q_idx, :]).unsqueeze(1) for q_idx in
             range(query_length)], 1)
        batch_pos_embeddings = torch.cat([single_pos_embeddings for _ in range(d_batch)], 0)
        return batch_pos_embeddings


class AlbertEmbedder(nn.Module):
    """
    Factorized embedder, as proposed in the ALBERT paper: http://arxiv.org/abs/1909.11942
    """
    def __init__(self,vocab):
        super().__init__()
        self.embedding_matrix = torch.rand(vocab.get_vocab_size('tokens'), FLAGS.d_emb)
        self.embedding_to_hidden = nn.Linear(FLAGS.d_emb, FLAGS.d_hidden)
    def forward(self, input):
        return self.embedding_to_hidden(self.embedding_matrix[input].cuda(FLAGS.device_idx))


def t5_denoise_spans_objective(tokens): # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683
    '''
    Produces inputs and targets.
    Inputs correspond to the original tokens, with a certain fraction of tokens replaced by a MASK-token.
    Contiguous tokens that happen to get masked get replaced by a single MASK token.
    There is no switching with random words, ... ( “MASS-style” objective )
    Targets look like: [mask_0, *<first word that was masked, possibly multiple if contiguous>, mask_1, <same>, ... , mask_eos, padding, padding, ... >
    '''
    #TODO maybe especially important to apply different mask every epoch? (if small text)
    masked_indices = sorted(random.sample(range(len(tokens)),int(len(tokens)*FLAGS.masking_fraction))) #

    given = [t if (i not in masked_indices) else MASKING_TOKEN for i, t in enumerate(tokens)]
    masked_given = [i for i, j in zip(given[1:], given[:-1]) if not (i == MASKING_TOKEN and i == j)]
    mask_counter = itertools.count()
    unique_masked_given = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_given]

    target = [tokens[i] for i in masked_indices]
    include_mask = [True] + [((i - j) != 1) for i, j in zip(masked_indices[1:], masked_indices[:-1])]
    masks = [MASKING_TOKEN if x else TO_BE_DELETED_TOKEN for x in include_mask]
    masked_target = [i for j in zip(masks, target) for i in j if i != TO_BE_DELETED_TOKEN]
    mask_counter = itertools.count() #Restart count
    unique_masked_target = [Token(f'{i}_{next(mask_counter)}') if i == MASKING_TOKEN else i for i in masked_target] + [Token(EOS_TOKEN)]
    return unique_masked_given, unique_masked_target