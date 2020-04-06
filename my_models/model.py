# %% Imports
from __future__ import unicode_literals, print_function

from operator import itemgetter

import math

from allennlp.data import Vocabulary
from allennlp.models import Model
import torch
import torch.nn as nn
from torch import tensor
from config import FLAGS, TOKENIZER_MAPPING
from constants import DECODER_START_TOKEN
from my_utils.model_utils import contrastive_L2_loss, apply_sequence_mask, process_targets_for_loss


class DIRTLMHead(Model):
    def __init__(self, finetune_stage=False):
        """
        finetune_stage: if True, don't do masking of words and internal activation
        """
        super().__init__(Vocabulary())
        self.embedder = AlbertEmbedder()
        self.encoder = MySequential(*[EncoderBlock(finetune_stage) for _ in range(FLAGS.nb_encoder_layers)])
        self.lm_head = LMHead()
        self.metrics_dict = {}
        self.finetune_stage = finetune_stage

    def get_metrics(self, **kwargs):
        return self.metrics_dict.copy() # copy needed to avoid overlapping train and validation metrics

    def forward(self, input_ids, padding_mask, masked_lm_labels=None):

        # ENCODING
        embedded_inputs = self.embedder(input_ids)
        encoded, _, cum_layer_loss = self.encoder(MyDropout()(embedded_inputs), padding_mask)
        cum_layer_loss = cum_layer_loss / FLAGS.nb_encoder_layers  # Normalize layer loss by number of times it is calculated
        result_dict = {}
        result_dict['encoded_activations'] = encoded

        vocab_scores = self.lm_head(encoded)

        if (masked_lm_labels is not None) and (not self.finetune_stage):
            targets = process_targets_for_loss(masked_lm_labels)
            vocab_scores_contiguous = vocab_scores.contiguous().view(-1, TOKENIZER_MAPPING[FLAGS.tokenizer].vocab_size)
            MLM_loss = nn.CrossEntropyLoss()(vocab_scores_contiguous,
                                             targets)
            result_dict['loss'] = FLAGS.DIR_loss_fraction * cum_layer_loss + (
                    1 - FLAGS.DIR_loss_fraction) * MLM_loss if FLAGS.DIR else MLM_loss

            self.metrics_dict['crossentropy_loss'] = MLM_loss.item()
            self.metrics_dict['perplexity'] = torch.exp(MLM_loss).item()
            self.metrics_dict['DIR_loss'] = cum_layer_loss.item() if isinstance(cum_layer_loss,
                                                                                torch.Tensor) else cum_layer_loss
        result_dict['vocab_scores'] = vocab_scores

        return result_dict  # Dictionary format for AllenNLP trainer loop

    def decode(self, output_dict):
        '''
        Overrides the AllenNLP decode method.
        Returns an actual best guess, needed at inference time, based on the probability distribution produced by :forward:
        '''
        prediction = output_dict['prediction_distribution'].max(-1)[1]
        return {'prediction': prediction}


def layer_normalize(param):
    mean = torch.mean(param, -1)
    mean_expanded = mean.unsqueeze(-1).expand(*(mean.shape + (param.shape[-1],)))
    st_dev = torch.sqrt(torch.mean((param - mean_expanded) ** 2, -1))
    st_dev_expanded = st_dev.unsqueeze(-1).expand(*(mean.shape + (param.shape[-1],)))
    return (param - mean_expanded) / st_dev_expanded


class LMHead(nn.Module):
    """
    Outputs a probability distribution over the vocabulary given a sequence of hidden states
    """

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.LayerNorm = nn.LayerNorm(FLAGS.d_hidden)
        self.decoder = nn.Linear(FLAGS.d_hidden, TOKENIZER_MAPPING[FLAGS.tokenizer].vocab_size)

    def forward(self, hidden_states):
        return self.decoder(self.LayerNorm(self.dense(
            hidden_states)))  # TODO maybe make a factorized ALBERT-like de-embedder as well? also share weights with output_embeddings.weight = input_embeddings.weight


class MyDropout(nn.Dropout):
    def __init__(self):
        super().__init__()
        self.p = FLAGS.dropout_rate


class FeedForwardBlock(nn.Module):
    """
    Feedforward block that is in every transformer block
    """

    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(FLAGS.d_hidden, FLAGS.d_ff)
        self.activation = nn.ReLU()
        self.linear_out = nn.Linear(FLAGS.d_ff, FLAGS.d_hidden)
        self.LayerNorm = nn.LayerNorm(FLAGS.d_hidden)

    def forward(self, hidden_in):
        result = MyDropout()(self.linear_out(self.activation(self.linear_in(hidden_in))))
        return self.LayerNorm(result + MyDropout()(hidden_in))


class EncoderBlock(nn.Module):
    def __init__(self,finetune_stage=False):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(finetune_stage=finetune_stage)
        # Dropout after every feedforward layer
        self.feedforward = FeedForwardBlock()
        self.finetune_stage = finetune_stage
        if FLAGS.DIR == 'top_down':
            self.top_down_regressor = nn.Sequential(
                nn.Linear(FLAGS.d_hidden,FLAGS.d_ff),
                nn.Linear(FLAGS.d_ff,FLAGS.d_hidden),
            )

    def forward(self, in_state, padding_mask,
                cum_layer_loss=0):
        attention_output_dict = self.multihead_attention(in_state, in_state, padding_mask)
        att_out = attention_output_dict['activations']
        out_state = self.feedforward(att_out)

        # Top-down regression
        if not self.finetune_stage:
            if FLAGS.DIR == 'top_down': #TODO make sure to only do (internal) denoising task when pretraining
                masked_in_state, mask = apply_sequence_mask(in_state)
                masked_att_out = self.multihead_attention(masked_in_state, in_state, padding_mask)['activations']
                masked_out_state = self.feedforward(masked_att_out) #TODO should add activation? And should add sometimes-not-masking?
                predicted_in_state = self.top_down_regressor(masked_out_state)
                layer_loss = contrastive_L2_loss(in_state, predicted_in_state, mask)
            elif FLAGS.DIR == 'from_projection':
                layer_loss = attention_output_dict['layer_loss']
        else:
            layer_loss = 0
        return out_state, padding_mask, layer_loss + cum_layer_loss


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



class Anticipation(nn.Module):
    def __init__(self):
        super().__init__()
        d_head = int(FLAGS.d_hidden / FLAGS.nb_heads)
        self.projector = nn.Linear(3 * d_head,
                                   d_head)

    def forward(self, q, k, v, position_embeddings, original_state):
        d_batch, d_seq, d_hidden = q.shape[0], q.shape[1], q.shape[2]
        nb_heads = FLAGS.nb_heads
        d_head = int(d_hidden / nb_heads)

        stacked_q, stacked_k, stacked_v = [t.transpose(1, 2).reshape(d_batch * nb_heads, d_head, d_seq) for t in
                                           [q, k, v]]  # [d_batch x nb_heads, d_head, d_seq]
        anticipation_input = torch.cat((stacked_q, stacked_k, stacked_v),
                                       dim=1)  # [d_batch x nb_heads, 3 x d_head, d_seq]
        # Transpose: make sure d_head-dimension is at the last place for nn.Linear
        projected_input = self.projector(anticipation_input.transpose(1, 2)).transpose(1,2)  # [d_batch x nb_heads, d_head, d_seq]

        masked_pos_embeddings, mask = apply_sequence_mask(position_embeddings)
        predicted_state_stacked = projected_input.bmm(masked_pos_embeddings)  # [d_batch x nb_heads, d_head, d_seq]
        predicted_state = predicted_state_stacked.reshape(d_batch, d_hidden, d_seq).transpose(1, 2)

        return contrastive_L2_loss(original_state, predicted_state, mask)

class MultiHeadAttention(nn.Module):

    def __init__(self, use_causal_mask=False,finetune_stage=False):
        super().__init__()
        # Only one big projection matrix is used, instead of a small projection matrix per head.
        # This is possible because in this implementation, the number of heads always needs to be a divisor of d_hidden
        # If so, then the big matrix is the same as a concatenation of smaller, per-head matrices would be.
        self.project_q = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_k = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_v = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_o = nn.Linear(FLAGS.d_hidden,
                                   FLAGS.d_hidden)  # In line with og t5 code (although not obvious from paper): there
        # for if different d_head_hidden than (FLAGS.d_hidden // FLAGS.nb_heads), here that's not supported atm
        self.use_causal_mask = use_causal_mask
        self.relative_attention_bias = nn.Embedding(FLAGS.relative_attention_num_buckets, FLAGS.nb_heads)
        self.LayerNorm = torch.nn.LayerNorm(FLAGS.d_hidden)
        self.finetune_stage = finetune_stage

        if FLAGS.DIR == 'from_projection':
            self.anticipation = Anticipation()

    def get_attention_mask(self, padding_mask, d_batch, query_length, value_length):
        """
        Produces a mask that indicates which replacers not to pay attention to.
        Combines a causal mask (if any) and a padding mask (if any)
        """
        model_device = self.project_k.weight.device
        if self.use_causal_mask:
            causal_mask = tensor([[[0 if value_index <= query_index else -float('inf') for value_index in
                                    range(value_length)] for query_index in range(query_length)]] * (
                                         d_batch * FLAGS.nb_heads)).cuda(model_device)
        if padding_mask is None:
            padding_mask = torch.zeros(d_batch, value_length).cuda(model_device)
        else:
            padding_mask = torch.log(padding_mask.type(
                torch.float))  # Because we are masking before pushing through softmax: we need -inf to have ) after softmax, and 0 to have 1 after softmax
        reshaped_padding_mask = padding_mask[:, None, :].repeat(1, query_length, FLAGS.nb_heads).view(-1, query_length,
                                                                                                      value_length)
        attention_mask = reshaped_padding_mask + causal_mask if self.use_causal_mask else reshaped_padding_mask
        return attention_mask  # [d_batch*FLAGS.num_heads, query_length, value_length]

    def forward(self, replacees, replacers,
                padding_mask=None):  # Padding mask to make sure we don't attend to padded positions
        '''
        Performs multi-headed attention: replacing each of the replacees by a weighted combination of all of the (learned value projections of) replacers.
        The weights are determined by a combination of 1) relative distance between replacer and replacee
        and 2) similarity of the (learned query projection of) the replacee to the (learned key projection of) the replacer g_mask:
        '''
        result_dict = {}
        d_batch = replacees.shape[0]
        q = self.project_q(replacees)
        k = self.project_k(replacers)
        v = self.project_v(replacers)
        assert FLAGS.d_hidden % FLAGS.nb_heads == 0
        d_head_hidden = FLAGS.d_hidden // FLAGS.nb_heads
        query_length = replacees.shape[1]
        value_length = replacers.shape[1]

        # This reshaping slices the last dimension and stacks those slices along the first dimension
        # In the resulting first dimension, first come all slices from the first batch, then from the second, and so on
        # This is relevant for how to add the position embeddings: they are the same per batch, but not per slice
        q_multi_parts = q.contiguous().view(d_batch * FLAGS.nb_heads, query_length, d_head_hidden)
        k_multi_parts = k.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)
        v_multi_parts = v.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)

        att_weights = torch.bmm(q_multi_parts,
                                k_multi_parts.transpose(1, 2))  # shape [d_batch x nb_heads, query_length, value_length]

        pos_embeddings = self.select_pos_embeddings(query_length, value_length, d_batch)
        batch_pos_embeddings = pos_embeddings.repeat(d_batch, 1, 1)

        att_weights += batch_pos_embeddings

        attention_mask = self.get_attention_mask(padding_mask, d_batch, query_length, value_length)
        att_weights += attention_mask

        att_weights = nn.Softmax(dim=-1)(att_weights)
        att_weights = MyDropout()(att_weights)

        att_output_multi_parts = torch.bmm(att_weights,
                                           v_multi_parts)  # [d_batch*num_heads,query_length, d_head_hidden] from [d_batch*num_heads, query_length, value_length] x [d_batch*num_heads,value_length, d_head_hidden]
        att_output = att_output_multi_parts.contiguous().view(d_batch, query_length, FLAGS.d_hidden)
        att_output = self.project_o(att_output)
        result_dict['activations'] = self.LayerNorm(att_output + MyDropout()(replacees))  # Include skip-connection

        if FLAGS.DIR == 'from_projection' and (not self.finetune_stage):
            assert torch.equal(replacees, replacers), 'from_projection DIR only works with self-attention.'
            result_dict['layer_loss'] = self.anticipation(q, k, v, batch_pos_embeddings, replacees)

        return result_dict

    def select_pos_embeddings(self, query_length, value_length, d_batch):
        rel_pos_indices = tensor(
            [[q_idx - k_idx for k_idx in range(value_length)] for q_idx in range(query_length)]) # shape [nb_heads, query_length, value_length]
        bucket_idxs = MultiHeadAttention._relative_position_bucket(rel_pos_indices).cuda(self.relative_attention_bias.weight.device)
        pos_embeddings = self.relative_attention_bias(bucket_idxs).permute(2, 0, 1)
        return pos_embeddings

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
            replacers in the range [0, num_buckets)
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
        # This puts all replacers larger than max_distance in the bucket for biggest numbers
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        bucket_indices += torch.where(is_small, n, val_if_large)
        return bucket_indices


class AlbertEmbedder(nn.Module):
    """
    Factorized embedder, as proposed in the ALBERT paper: http://arxiv.org/abs/1909.11942
    """

    def __init__(self):
        super().__init__()
        self.idx_to_embedding = nn.Embedding(TOKENIZER_MAPPING[FLAGS.tokenizer].vocab_size, FLAGS.d_emb)
        self.embedding_to_hidden = nn.Linear(FLAGS.d_emb, FLAGS.d_hidden)
        self.LayerNorm = torch.nn.LayerNorm(FLAGS.d_hidden)

    def forward(self, idxs):
        embedded = self.idx_to_embedding(idxs)
        hidden = self.embedding_to_hidden(embedded)
        normalized_dropped_out_hidden = self.LayerNorm(MyDropout()(hidden))
        return normalized_dropped_out_hidden
