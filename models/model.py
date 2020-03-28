# %% Imports
from __future__ import unicode_literals, print_function

from operator import itemgetter

import math

from allennlp.models import Model
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import tensor
from config import FLAGS, TOKENIZER_MAPPING
from constants import DECODER_START_TOKEN
from util import masked_MSE_loss


class FullModel(Model):
    def __init__(self, vocab, do_teacher_forcing=True):
        """
        """
        super().__init__(vocab)
        self.embedder = AlbertEmbedder()
        self.encoder = MySequential(*[EncoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        if FLAGS.use_decoder:
            self.decoder = MySequential(
                *[DecoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.predictor = Predictor()
        self.teacher_forcing = do_teacher_forcing
        self.metrics_dict = {}

    def get_metrics(self, **kwargs):
        return self.metrics_dict

    def process_targets_for_loss(self, target_tokens,
                                 max_target_seq_length):
        padding_index = 0
        current_target_seq_length = target_tokens.shape[1]
        padder = nn.ConstantPad1d((0, max_target_seq_length - current_target_seq_length), padding_index)
        target_tokens_contiguous = padder(target_tokens).contiguous().view(-1)

        return target_tokens_contiguous

    def decode_idxs_to_probabilities(self, decoder_input_token_idxs, encoded, padding_mask):
        decoder_input_embeddings, _ = self.embedder(decoder_input_token_idxs)
        decoded, _, _, cum_layer_loss = self.decoder(nn.Dropout()(decoder_input_embeddings), encoded, padding_mask)
        vocab_scores = self.predictor(decoded)
        return vocab_scores, cum_layer_loss

    def forward(self, masked_ids, padding_mask, masked_lm_labels=None):
        d_batch = masked_ids.shape[
            0]  # Actual batch size (might not equal FLAGS.d_batch, eg when not enough samples to fill the last batch
        max_target_seq_length = int(FLAGS.max_seq_length * FLAGS.masking_fraction * 2 + 1) if (
                masked_lm_labels is None) else masked_lm_labels.shape[-1]  # Longest length if no adjacent masks
        targets = self.process_targets_for_loss(masked_lm_labels, max_target_seq_length)

        # ENCODING
        embedded_inputs = self.embedder(masked_ids)
        encoded, _, cum_layer_loss = self.encoder(MyDropout()(embedded_inputs), padding_mask)
        cum_layer_loss = cum_layer_loss / FLAGS.nb_encoder_layers  # Normalize layer loss by number of times it is calculated
        result_dict = {}
        # DECODING
        if FLAGS.use_decoder:  # TODO DECODER: maybe add some assert that checks if targets (if any) are in the correct format
            if self.teacher_forcing:
                # With teacher forcing, we can parallelize decoding using a causal mask
                shifted_target_tokens = torch.cat(
                    (tensor([[self.vocab.get_token_index(DECODER_START_TOKEN)]] * d_batch).cuda(),
                     # TODO DECODER: change this to not use self.vocab, but directly an id
                     masked_lm_labels[:, :-1]),
                    dim=1)  # Teacher forcing: shift to the right by one (add 'start' token in front, and drop last token as not used anyway)
                vocab_scores, cum_decoder_layer_loss = self.decode_idxs_to_probabilities(shifted_target_tokens, encoded,
                                                                                         padding_mask)
                cum_decoder_layer_loss /= 2 * FLAGS.nb_decoder_layers  # Normalize layer loss by number of times it is calculated
                cum_layer_loss = (cum_layer_loss + cum_decoder_layer_loss) / 2
                _, output_idxs = torch.max(vocab_scores, dim=-1)
            else:
                vocab_scores, output_idxs = self.beam_decode(d_batch, encoded, max_target_seq_length, padding_mask)
                result_dict['output_idxs'] = output_idxs
        else:
            vocab_scores = self.predictor(encoded)

        if targets is not None:
            vocab_scores_contiguous = vocab_scores.contiguous().view(-1, TOKENIZER_MAPPING[FLAGS.tokenizer].vocab_size)
            MLM_loss = nn.CrossEntropyLoss()(vocab_scores_contiguous,
                                             targets)
            result_dict['loss'] = FLAGS.DIR_loss_fraction * cum_layer_loss + (
                    1 - FLAGS.DIR_loss_fraction) * MLM_loss if FLAGS.use_DIR else MLM_loss

            self.metrics_dict['crossentropy_loss'] = MLM_loss.item() if isinstance(MLM_loss, torch.Tensor) else MLM_loss
            self.metrics_dict['DIR_loss'] = cum_layer_loss.item() if isinstance(cum_layer_loss,
                                                                                torch.Tensor) else cum_layer_loss
        result_dict['vocab_scores'] = vocab_scores

        return result_dict  # Dictionary format for AllenNLP trainer loop

    def beam_decode(self, d_batch, encoded, max_target_seq_length,
                    padding_mask):  # TODO DECODER: make it so that this also outputs vocab probabilities, to allow training without teacher forcing
        k = FLAGS.beam_width
        out_seq_length = max_target_seq_length + 1  # To allow start token in the output
        output_idxs = torch.zeros(d_batch, out_seq_length, k, dtype=torch.long).cuda()
        output_idxs[:, 0, :] = self.vocab.get_token_index(
            DECODER_START_TOKEN)  # TODO DECODER: change this to not use self.vocab, but directly an id
        output_probs = torch.ones(d_batch, k, dtype=torch.long).to(
            FLAGS.device_idx)  # starting probability for product of probabilities along path
        for i in range(max_target_seq_length):  # TODO DECODER: this takes pretty long :P find fix?

            # Stack decoder inputs for different candidates in the beam along batch dimension, so they can be
            # processed in parallel
            stacked_output_idxs = output_idxs.permute(0, 2, 1).reshape(d_batch * k,
                                                                       out_seq_length)  # Stacks in batch dim as follows: [b1k1, b1k2, b2k1,b2k2, b3k1, b3k2] if d_batch = 3 and k = 2
            stacked_encoded, stacked_padding_mask = encoded.repeat(k, 1, 1), padding_mask.repeat(k, 1)
            stacked_vocab_probs = self.decode_idxs_to_probabilities(stacked_output_idxs, stacked_encoded,
                                                                    stacked_padding_mask)  # [d_batch*k (stacked as explained above), max_seq_length, vocab_size]

            # Unstack and get result for current sequence element
            vocab_probs = stacked_vocab_probs.reshape(d_batch, k, out_seq_length, -1).permute(0, 2, 1,
                                                                                              3)  # [d_batch, out_seq_length, k, vocab_size]
            current_vocab_probs = vocab_probs[:, i, :, :]  # [d_batch, k, vocab_size]

            # Get the top k probabilities of each word in the vocab for each of the k current top paths in beam search
            top_kk_probs, top_kk_idxs = torch.topk(current_vocab_probs, k=k,
                                                   dim=-1)  # [d_batch,k_prev (each previous index), k_now (top k indices for each previous index)]

            top_kk_path_probs = top_kk_probs * output_probs[:, :, None].repeat(1, 1, k)

            # Get the top k next path probabilities, over each previous path
            top_k_path_probs, top_k_meta_idxs = torch.topk(top_kk_path_probs.reshape(d_batch, k * k), k=k, dim=-1)
            top_k_path_probs /= top_k_path_probs.max(-1)[0][:, None].repeat(1,
                                                                            k)  # Scaling path probabilities to avoid underflow. Scaling such that highest probability for each sample is 1
            top_k_prev_meta_idxs = top_k_meta_idxs // k  # Get the previous paths to keep
            top_k_now_idxs = torch.gather(top_kk_idxs.reshape(d_batch, k * k), -1, top_k_meta_idxs)
            output_idxs = torch.gather(output_idxs, -1, top_k_prev_meta_idxs[:, None, :].repeat(1, out_seq_length,
                                                                                                1))  # replace paths with paths that give the top probs now
            output_vocab_probs = torch.gather(output_vocab_probs, -1,
                                              top_k_prev_meta_idxs[:, None, :].repeat(1, out_seq_length,
                                                                                      1))
            output_idxs[:, i + 1, :] = top_k_now_idxs
            output_vocab_probs[:, i + 1, :] = current_vocab_probs
            output_probs = top_k_path_probs
        return output_idxs[:, 1:, 0]

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


class Predictor(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiHeadAttention()
        # Dropout after every feedforward layer
        self.feedforward = FeedForwardBlock()

    def forward(self, input, padding_mask,
                cum_layer_loss=0):
        att_out, layer_loss = itemgetter('activations', 'layer_loss')(
            self.multihead_attention(input, input, padding_mask))
        ff_out = self.feedforward(att_out)
        return ff_out, padding_mask, layer_loss + cum_layer_loss


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
        self.attention = MultiHeadAttention()
        # Dropout after every feedforward layer
        self.feedforward = FeedForwardBlock()

    def forward(self, output, original_encoded_input, padding_mask):
        self_att_out, layer_loss_1 = itemgetter('activations', 'layer_loss')(self.self_attention(query=output,
                                                                                                 values=output))  # Include skip-connection and layer normalization
        att_out, layer_loss_2 = itemgetter('activations', 'layer_loss')(
            self.attention(query=self_att_out, values=original_encoded_input, padding_mask=padding_mask))
        ff_out = self.feedforward(att_out)
        return ff_out, original_encoded_input, padding_mask, layer_loss_1 + layer_loss_2


class Anticipation(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Linear(3 * FLAGS.max_seq_length,
                                   FLAGS.max_seq_length)
    def forward(self, projected_q, projected_k, projected_v, position_embeddings, original_query):
        mask = (torch.rand(projected_q.shape[1]) > FLAGS.masking_fraction).cuda(
            projected_q.device)  # Same mask for items in batch
        broadcast_ready_mask = mask[None, :, None].repeat(1,3,1)
        anticipation_input = torch.cat((projected_q, projected_v, projected_k), dim=1)
        masked_anticipation_input = anticipation_input * broadcast_ready_mask
        predicted_query = self.regressor(masked_anticipation_input.permute(0, 2, 1)).permute(0, 2, 1)
        if FLAGS.d_batch <= 1:
            raise ValueError('Using DIR requires batch size bigger than 1 to contrast with')
        actual_batch_size = original_query.shape[0]  # Might differ eg at the end of the data
        # Negative loss formed by distances to other batch elements AND correct batch element
        negative_loss = sum([masked_MSE_loss(original_query.roll(shifts=i, dims=0), predicted_query, mask) for i in
                             range(actual_batch_size)]) / actual_batch_size if actual_batch_size > 1 else torch.tensor(
            1.)

        # Positive loss: distance to corresponding batch element
        positive_loss = masked_MSE_loss(original_query, predicted_query, mask)

        total_loss = positive_loss / negative_loss
        return total_loss


class MultiHeadAttention(nn.Module):

    def __init__(self, use_causal_mask=False):
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
        if FLAGS.use_DIR:
            self.anticipation = Anticipation()

    def get_attention_mask(self, padding_mask, d_batch, query_length, value_length):
        """
        Produces a mask that indicates which values not to pay attention to.
        Combines a causal mask (if any) and a padding mask (if any)
        """

        if self.use_causal_mask:
            causal_mask = tensor([[[0 if value_index <= query_index else -float('inf') for value_index in
                                    range(value_length)] for query_index in range(query_length)]] * (
                                         d_batch * FLAGS.nb_heads)).cuda()
        if padding_mask is None:
            padding_mask = torch.zeros(d_batch, value_length).cuda()
        else:
            padding_mask = torch.log(padding_mask.type(
                torch.float))  # Because we are masking before pushing through softmax: we need -inf to have ) after softmax, and 0 to have 1 after softmax
        reshaped_padding_mask = padding_mask[:, None, :].repeat(1, query_length, FLAGS.nb_heads).view(-1, query_length,
                                                                                                      value_length)
        attention_mask = reshaped_padding_mask + causal_mask if self.use_causal_mask else reshaped_padding_mask
        return attention_mask  # [d_batch*FLAGS.num_heads, query_length, value_length]

    def forward(self, query, values,
                padding_mask=None):  # Padding mask to make sure we don't attend to padded positions
        result_dict = {}
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
        result_dict['activations'] = self.LayerNorm(att_output + MyDropout()(query))  # Include skip-connection

        result_dict['layer_loss'] = self.anticipation(q, k, v, pos_embeddings, query) if FLAGS.use_DIR else 0
        return result_dict

    def select_pos_embeddings(self, query_length, value_length, d_batch):
        rel_pos_indices = tensor(
            [[q_idx - k_idx for k_idx in range(value_length)] for q_idx in range(query_length)]) \
            .cuda()  # shape [nb_heads, query_length, value_length]
        bucket_idxs = MultiHeadAttention._relative_position_bucket(rel_pos_indices)
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
        # This puts all values larger than max_distance in the bucket for biggest numbers
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
