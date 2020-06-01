# %% Imports
from __future__ import unicode_literals, print_function

from collections import OrderedDict

import re
from operator import itemgetter

import math

from allennlp.data import Vocabulary
from allennlp.models import Model
import torch
import torch.nn as nn
from torch import tensor
from transformers import AlbertModel, AlbertForMaskedLM

from config import FLAGS, get_my_tokenizer
from constants import TYPE_VOCAB_SIZE
from my_utils.model_utils import contrastive_L2_loss, apply_sequence_mask, process_targets_for_loss, get_activation, \
    sizeof_fmt, sz
import logging as log
log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa


class DIRTLMHead(Model):
    def __init__(self, finetune_stage=False):
        """
        finetune_stage: if True, don't do masking of words and internal activation
        """
        super().__init__(Vocabulary())
        self.embedder = AlbertEmbedder()
        self.shared_encoder_block = EncoderBlock(finetune_stage)
        if FLAGS.DIR == 'combo':
            self.combiner = nn.Linear(3*FLAGS.d_hidden,FLAGS.d_hidden)
            self.shared_top_down_predictor = nn.Sequential(
                nn.Linear(FLAGS.d_hidden, FLAGS.d_ff),
                get_activation(),
                nn.Linear(FLAGS.d_ff, FLAGS.d_hidden),
            )
            self.shared_from_left_predictor = nn.Sequential(
                nn.Linear(FLAGS.d_hidden, FLAGS.d_ff),
                get_activation(),
                nn.Linear(FLAGS.d_ff, FLAGS.d_hidden),
            )
            self.shared_from_right_predictor = nn.Sequential(
                nn.Linear(FLAGS.d_hidden, FLAGS.d_ff),
                get_activation(),
                nn.Linear(FLAGS.d_ff, FLAGS.d_hidden),
            )
            self.learn_phase = True
        self.lm_head = LMHead()
        self.metrics_dict = {}
        self.finetune_stage = finetune_stage
        self.dropout = MyDropout()

        if FLAGS.use_HFpretrained_weights:
            self.load_HFpretrained_weights()

    def load_HFpretrained_weights(self):
        hf_state_dict = AlbertForMaskedLM.from_pretrained(FLAGS.hf_model_handle).state_dict()
        repl = {"albert.embeddings": 'embedder',
                'word_embeddings':'idx_to_embedding',
                'albert.encoder.embedding_hidden_mapping_in': 'embedder.embedding_to_hidden',
                'albert.encoder.albert_layer_groups.0.albert_layers.0': 'shared_encoder_block',
                'attention.dense': 'multihead_attention.project_o',
                'attention': 'multihead_attention',
                'full_layer_layer_norm':'feedforward.LayerNorm',
                'query': 'project_q',
                'key': 'project_k',
                'value': 'project_v',
                'ffn.': 'feedforward.linear_in.',
                'ffn_output': 'feedforward.linear_out',
                'predictions': 'lm_head', }
        # use these three lines to do the replacement
        repl = dict((re.escape(k), v) for k, v in repl.items())
        pattern = re.compile("|".join(repl.keys()))
        updated_hf_state_dict = OrderedDict(
            (pattern.sub(lambda m: repl[re.escape(m.group(0))], k), v) for k, v in hf_state_dict.items())
        # Allow for cutting the sequence length short
        updated_hf_state_dict['embedder.position_embeddings.weight'] = updated_hf_state_dict[
                                                                           'embedder.position_embeddings.weight'][
                                                                       :FLAGS.max_seq_length, :].clone()
        missing, unexpected = self.load_state_dict(updated_hf_state_dict,strict=False)
        # Allowed discrepancies: don't care about pooler, and have optional relative attention bias, + there is a 'lm_head.bias' that is only used to set lm head decoder bias to zero, which I' currently ignoring :P
        ignored_hf_parameters = [
            'pooler',
            'position_embeddings',
            'lm_head.bias']
        allowed_from_scratch_params = [
            'relative_attention_bias',
            'top_down_regressor',
            'combiner','shared_top_down_predictor','shared_from_left_predictor','shared_from_right_predictor'

        ]
        for m in missing:
            if not any([s in m for s in allowed_from_scratch_params]):
                raise ValueError(f'Unexpected mismatch in loading state dict: {m} not present in pretrained.')
        for u in unexpected:
            if not any([s in u for s in ignored_hf_parameters]):
                raise ValueError(f'Unexpected mismatch in loading state dict: {u} in pretrained but not in current model.')
        log.info(f"Loaded pretrained weights from {FLAGS.hf_model_handle}")

    def get_metrics(self, **kwargs):
        return self.metrics_dict.copy() # copy needed to avoid overlapping train and validation metrics

    def forward(self, input_ids, padding_mask, masked_lm_labels=None,token_type_ids=None):

        # ENCODING
        clean = (FLAGS.DIR != 'combo') or self.finetune_stage
        if FLAGS.DIR == 'combo':
            normalizer = FLAGS.nb_encoder_layers - FLAGS.top_down_distance

            if FLAGS.alternate_internal_prediction:
                self.learn_phase = not self.learn_phase
        else:
            normalizer = FLAGS.nb_encoder_layers
        if clean:
            encoder = MySequential(*[self.shared_encoder_block for _ in range(FLAGS.nb_encoder_layers)],clean=clean)
        else:

            encoder = MySequential(*[self.shared_encoder_block for _ in range(FLAGS.nb_encoder_layers)],
                               top_down=self.shared_top_down_predictor,
                               from_left=self.shared_from_left_predictor,
                               from_right=self.shared_from_right_predictor,
                               combiner=self.combiner,
                               clean=clean,
                                   learn_phase=self.learn_phase)
        embedded_inputs = self.embedder(input_ids,token_type_ids)
        encoded, _, cum_layer_loss, layer_loss_list = encoder(embedded_inputs, padding_mask)

        cum_layer_loss = cum_layer_loss / normalizer  # Normalize layer loss by number of times it is calculated
        result_dict = {}
        result_dict['encoded_activations'] = encoded

        vocab_scores = self.lm_head(encoded)

        if (masked_lm_labels is not None) and (not self.finetune_stage):
            targets = process_targets_for_loss(masked_lm_labels)
            vocab_scores_contiguous = vocab_scores.contiguous().view(-1, get_my_tokenizer().vocab_size)
            MLM_loss = nn.CrossEntropyLoss()(vocab_scores_contiguous,
                                             targets)
            result_dict['loss'] = FLAGS.DIR_loss_fraction * cum_layer_loss + (
                    1 - FLAGS.DIR_loss_fraction) * MLM_loss if FLAGS.DIR else MLM_loss

            self.metrics_dict['crossentropy_loss'] = MLM_loss.item()
            self.metrics_dict['perplexity'] = torch.exp(MLM_loss).item()

            if FLAGS.DIR and self.learn_phase:
                self.metrics_dict['DIR_loss'] = cum_layer_loss.item() if isinstance(cum_layer_loss,
                                                                                    torch.Tensor) else cum_layer_loss
                for layer, loss in enumerate(layer_loss_list):
                    self.metrics_dict[f'DIR_loss_layer_{layer}'] = loss.item() if isinstance(loss,
                                                                                torch.Tensor) else loss

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
        self.dense = nn.Linear(FLAGS.d_hidden, FLAGS.d_emb)
        self.LayerNorm = nn.LayerNorm(FLAGS.d_emb)
        self.decoder = nn.Linear(FLAGS.d_emb, get_my_tokenizer().vocab_size) #TODO add activation here for consistency with ALBERT
        self.activation = get_activation()

    def forward(self, hidden_states):
        return self.decoder(self.LayerNorm(self.activation(self.dense(hidden_states))))


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
        self.activation = get_activation()
        self.linear_out = nn.Linear(FLAGS.d_ff, FLAGS.d_hidden)
        self.LayerNorm = InternalLayerNorm(FLAGS.d_hidden)

    def forward(self, hidden_in):
        result = self.linear_out(self.activation(self.linear_in(hidden_in)))
        return self.LayerNorm(result + hidden_in)


class EncoderBlock(nn.Module):
    def __init__(self,finetune_stage=False):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(finetune_stage=finetune_stage)
        self.feedforward = FeedForwardBlock()
        self.finetune_stage = finetune_stage
        if FLAGS.DIR == 'top_down':
            self.top_down_regressor = nn.Sequential(
                nn.Linear(FLAGS.d_hidden,FLAGS.d_top_down),
                nn.Linear(FLAGS.d_top_down,FLAGS.d_hidden),
            )
            # self.top_down_regressor = nn.Sequential()
    def forward(self, in_state, padding_mask,
                cum_layer_loss=0,layer_loss_list=[]):
        attention_output_dict = self.multihead_attention(in_state, in_state, padding_mask)
        att_out = attention_output_dict['activations']
        out_state = self.feedforward(att_out)

        # Top-down regression
        if not self.finetune_stage: #TODO make sure it doesn't use all the extra mem here
            if FLAGS.DIR == 'top_down':
                masked_in_state, mask = apply_sequence_mask(in_state)
                masked_att_out = self.multihead_attention(masked_in_state, in_state, padding_mask)['activations']
                masked_out_state = self.feedforward(masked_att_out) #TODO should add activation? And should add sometimes-not-masking?
                predicted_in_state = self.top_down_regressor(masked_out_state)
                layer_loss = contrastive_L2_loss(in_state, predicted_in_state, mask)
            elif FLAGS.DIR == 'from_projection':
                layer_loss = attention_output_dict['layer_loss']
            else:
                layer_loss = 0
        else:
            layer_loss = 0
        layer_loss_list.append(layer_loss)
        return out_state, padding_mask, layer_loss + cum_layer_loss, layer_loss_list


class MySequential(nn.Sequential): #TODO move this to a for loop in enclosing module
    '''
    Allows Sequential to pass on multiple in- and outputs
    '''
    def __init__(self,*args,top_down=None,from_left=None,from_right=None,combiner=None,clean=True,learn_phase=True):
        super().__init__(*args)
        self.clean = clean
        self.learn_phase = learn_phase
        if not self.clean:
            self.top_down_predictor = top_down
            self.from_left_predictor = from_left
            self.from_right_predictor = from_right
            self.combiner = combiner

    def forward(self, *inputs):
        layers = self[:FLAGS.nb_encoder_layers]
        cum_layer_loss = 0
        layer_loss_list = []
        for layer_idx, module in enumerate(layers): #TODO fix heavy mem overhead
            if FLAGS.DIR == 'combo' and (layer_idx + FLAGS.top_down_distance < len(layers)) and (not self.clean):
                in_activations, padding_mask = inputs[:2]
                masked_inputs, DIRT_mask = apply_sequence_mask(in_activations)
                contextualizer = layers[layer_idx:layer_idx + FLAGS.top_down_distance] #Clean true by default
                top_down_inputs,_, _,_ = contextualizer(masked_inputs,padding_mask)
                left_adjacent_inputs = in_activations.roll(shifts=1,dims=1)
                right_adjacent_inputs = in_activations.roll(shifts=-1,dims=1)
                top_down_prediction = self.top_down_predictor(top_down_inputs) # #TODO maybe add some dropout in dirt parts
                from_left_prediction = self.from_left_predictor(left_adjacent_inputs)
                from_right_prediction = self.from_right_predictor(right_adjacent_inputs)
                combined_prediction = self.combiner(torch.cat((top_down_prediction,from_left_prediction,from_right_prediction),
                                                              dim=-1))
                # The first and last sequence elements don't have proper left resp. right inputs.
                # Don't consider these in calculating the loss
                edge_mask = torch.zeros_like(DIRT_mask)
                edge_mask[0] = True
                edge_mask[-1] = True
                DIRT_mask = DIRT_mask | edge_mask
                if self.learn_phase:
                    layer_loss = contrastive_L2_loss(in_activations, combined_prediction, DIRT_mask)
                    cum_layer_loss += layer_loss
                    layer_loss_list.append(layer_loss)
                else:
                    inputs = (torch.where(DIRT_mask[None,:,None],inputs[0],combined_prediction),) + inputs[1:]




            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        if (not self.clean) and FLAGS.DIR == 'combo':
            inputs = inputs[:2] +(cum_layer_loss,) + (layer_loss_list,)
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
        if FLAGS.pos_embeddings == 'relative':
            self.relative_attention_bias = nn.Embedding(FLAGS.relative_attention_num_buckets, FLAGS.nb_heads)
        self.LayerNorm = InternalLayerNorm(FLAGS.d_hidden)
        self.finetune_stage = finetune_stage
        self.dropout = MyDropout()
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
        q_multi_parts = q.transpose(1,2).contiguous().view(d_batch * FLAGS.nb_heads, d_head_hidden, query_length).transpose(1,2) #Transpose: so each head has the first slice _of every sequence element_
        k_multi_parts = k.transpose(1,2).contiguous().view(d_batch * FLAGS.nb_heads, d_head_hidden, value_length).transpose(1,2)
        v_multi_parts = v.transpose(1,2).contiguous().view(d_batch * FLAGS.nb_heads, d_head_hidden, value_length).transpose(1,2)

        att_weights = torch.bmm(q_multi_parts,
                                k_multi_parts.transpose(1, 2))  # shape [d_batch x nb_heads, query_length, value_length]
        att_weights /= math.sqrt(d_head_hidden) # Scaling to be in line with Albert (even if t5 didn't do this)
        if FLAGS.pos_embeddings == 'relative':
            pos_embeddings = self.select_pos_embeddings(query_length, value_length)
            batch_pos_embeddings = pos_embeddings.repeat(d_batch, 1, 1)

            att_weights += batch_pos_embeddings

        att_weights += self.get_attention_mask(padding_mask, d_batch, query_length, value_length) #adding attention mask

        att_weights = nn.Softmax(dim=-1)(att_weights)
        att_weights = self.dropout(att_weights)

        att_output_multi_parts = torch.bmm(att_weights,
                                           v_multi_parts)  # [d_batch*num_heads,query_length, d_head_hidden] from [d_batch*num_heads, query_length, value_length] x [d_batch*num_heads,value_length, d_head_hidden]
        att_output = att_output_multi_parts.transpose(1,2).contiguous().view(d_batch, FLAGS.d_hidden, query_length).transpose(1,2).contiguous() # Last contiguous to make sure mem calculations add up :P
        att_output = self.project_o(att_output) # Ok THIS I did better than HF :D
        result_dict['activations'] = self.LayerNorm(self.dropout(att_output) + replacees)  # Include skip-connection

        if FLAGS.DIR == 'from_projection' and (not self.finetune_stage):
            assert torch.equal(replacees, replacers), 'from_projection DIR only works with self-attention.'
            result_dict['layer_loss'] = self.anticipation(q, k, v, batch_pos_embeddings, replacees)

        return result_dict

    def select_pos_embeddings(self, query_length, value_length):
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

class InternalLayerNorm(torch.nn.LayerNorm):
    # To be in accordance with HF Albert
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.eps = FLAGS.layernorm_eps
class AlbertEmbedder(nn.Module):
    """
    Factorized embedder, as proposed in the ALBERT paper: http://arxiv.org/abs/1909.11942
    """

    def __init__(self):
        super().__init__()
        self.idx_to_embedding = nn.Embedding(get_my_tokenizer().vocab_size, FLAGS.d_emb)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, FLAGS.d_emb)

        if FLAGS.pos_embeddings == 'absolute':
            self.position_embeddings = nn.Embedding(FLAGS.max_seq_length, FLAGS.d_emb)
        self.embedding_to_hidden = nn.Linear(FLAGS.d_emb, FLAGS.d_hidden)
        self.LayerNorm = InternalLayerNorm(FLAGS.d_emb)
        self.dropout = MyDropout()
    def forward(self, ids,token_type_ids=None):
        embedded = self.idx_to_embedding(ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(ids)
        token_embedded = self.token_type_embeddings(token_type_ids)
        pos_embedded = torch.zeros_like(embedded)
        if FLAGS.pos_embeddings == 'absolute':
            pos_idxs = torch.arange(ids.shape[1], device=ids.device)[None, :].expand(ids.shape)
            pos_embedded = self.position_embeddings(pos_idxs)
        total_embedded = embedded + token_embedded + pos_embedded
        normalized_embedded = self.LayerNorm(total_embedded)
        dropped_out_embedded = self.dropout(normalized_embedded)
        hidden = self.embedding_to_hidden(dropped_out_embedded)
        return hidden
