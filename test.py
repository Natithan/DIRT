# %% Imports
from __future__ import unicode_literals, print_function

import os
import sys
from functools import reduce
from pathlib import Path
from typing import Any, T_co

from allennlp.models import Model
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
from spacy.lang.en import English  # updated
import matplotlib
import matplotlib.pyplot as plt
from allennlp.data import DatasetReader, Instance, Token, Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
import operator
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools

from absl import app
from absl import flags

# %% FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 3, "Batch size")
flags.DEFINE_integer("d_emb", 12, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 72, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
# flags.DEFINE_integer("target_length", 20, "Number of tokens in target sequence")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")
# flags.DEFINE_integer("source_length", 20, "Number of tokens in source sequence")
flags.DEFINE_integer("max_seq_length", 20, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", "./data/Gutenberg", "Folder with train, val and test subfolders containing data")

# Trainer flags
flags.DEFINE_integer("patience", 10, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 1000, "Number of epochs to train for.")

flags.DEFINE_bool("mini", True, "Whether to work with mini data/models for debugging purposes")

flags.DEFINE_integer("nb_encoder_layers", 2, "Number of layers in the encoder.")
flags.DEFINE_integer("nb_decoder_layers", 2, "Number of layers in the decoder.")
flags.DEFINE_integer("nb_feedforward_layers", 2, "Number of layers in the feedforward subcomponents of the transformer.")
FLAGS(sys.argv)
# %%

def t5_denoise_spans_objective(tokens): # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683
    masked_indices = sorted(random.sample(range(len(tokens)),int(len(tokens)*FLAGS.masking_fraction)))

    given = [t if (i not in masked_indices) else '@@MASK@@' for i, t in enumerate(tokens)]
    masked_given = [i for i, j in zip(given[1:], given[:-1]) if not (i == '@@MASK@@' and i == j)]
    mask_counter = itertools.count()
    unique_masked_given = [Token(f'{i}_{next(mask_counter)}') if i == '@@MASK@@' else i for i in masked_given]

    target = [tokens[i] for i in masked_indices]
    include_mask = [True] + [((i - j) != 1) for i, j in zip(masked_indices[1:], masked_indices[:-1])]
    masks = ['@@MASK@@' if x else '@@TO_BE_DELETED@@' for x in include_mask]
    masked_target = [i for j in zip(masks, target) for i in j if i != '@@TO_BE_DELETED@@']
    mask_counter = itertools.count() #Restart count
    unique_masked_target = [Token(f'{i}_{next(mask_counter)}') if i == '@@MASK@@' else i for i in masked_target] + [Token('@@MASK_EOS@@')]
    return unique_masked_given, unique_masked_target


class GutenbergReader(DatasetReader):

    def __init__(self, token_indexers=None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens, tags=None):
        inputs, targets = t5_denoise_spans_objective(tokens)
        input_field = TextField(inputs, self.token_indexers)
        target_field = TextField(targets, self.token_indexers)
        fields = {"inputs": input_field,
                  "targets": target_field}

        # if tags:
        #     label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        #     fields["labels"] = label_field

        return Instance(fields)

    def _read(self, folder_path):
        for i, file in enumerate(os.scandir(folder_path)):
            if FLAGS.mini:
                if i > 0:
                    break
            with open(file) as f:
                running_sequence = []
                for j, line in enumerate(f):
                    if FLAGS.mini:
                        if j > 50:
                            break
                    words = line.strip().split()
                    running_sequence += words
                    if len(running_sequence) >= FLAGS.max_seq_length:
                        current_sequence = running_sequence[:FLAGS.max_seq_length]
                        running_sequence = running_sequence[FLAGS.max_seq_length:]
                        yield self.text_to_instance([Token(word) for word in current_sequence])



class FullModel(Model):
    def __init__(self,vocab):
        """
        """
        super().__init__(vocab)
        self.vocab = vocab
        self.embedder = AlbertEmbedder(vocab)
        self.encoder = nn.Sequential(*[EncoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.decoder = MySequential(*[DecoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.predictor = nn.Linear(FLAGS.d_hidden, self.vocab.get_vocab_size())
        self.loss = nn.CrossEntropyLoss()

    def get_parameters_for_histogram_tensorboard_logging(self):
        return [self.loss]

    def process_targets_for_loss(self, targets,max_target_seq_length):
        target_tokens = targets['tokens']
        padding_index = self.vocab.get_token_index('@@PAD@@')
        current_target_seq_length = target_tokens.shape[1]
        padder = nn.ConstantPad1d((0, max_target_seq_length - current_target_seq_length), padding_index)
        target_tokens_contiguous = padder(target_tokens).contiguous().view(-1)

        return target_tokens_contiguous

    def forward(self, inputs, targets):
        d_batch = inputs['tokens'].shape[0] # Actual batch size (might not equal FLAGS.d_batch, eg when not enough samples to fill the last batch
        max_target_seq_length = int(FLAGS.max_seq_length*FLAGS.masking_fraction*2 + 1) # Longest length if no adjacent masks
        targets = self.process_targets_for_loss(targets,max_target_seq_length)
        embedded_inputs = self.embedder(inputs['tokens'])
        embedded_outputs = torch.rand(d_batch, max_target_seq_length,FLAGS.d_hidden).cuda()
        encoded = self.encoder(nn.Dropout()(embedded_inputs))
        decoded,_ = self.decoder(nn.Dropout()(embedded_outputs),encoded)
        prediction_distribution = nn.Softmax()(self.predictor(nn.Dropout()(decoded))) #TODO add explicit dimension arg to softmax to avoid warning
        prediction_distribution_contiguous = prediction_distribution.contiguous().view(-1,self.vocab.get_vocab_size())
        prediction = prediction_distribution.max(-1)[1]
        loss =  self.loss(prediction_distribution_contiguous,targets) #TODO figure out why performance barely improving :P
        return {'loss':loss} # For AllenNLP trainer loop #TODO figure out error during validation time



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
        att_out = layer_normalize(self.attention(query=self_att_out, values=original_encoded_input )) # TODO make sure values match in batch dimension
        ff_out = layer_normalize(self.feedforward(att_out) + nn.Dropout()(att_out))
        return ff_out, original_encoded_input

class MultiHeadAttention(nn.Module):
    # relative position embedding with d_emb=1 (aka a scalar), shared across layers, but different between attention heads, in line with t5 paper
    # nb of possible relative positions ranges from - max_seq_length to + max_seq_length
    position_embedding = torch.rand(FLAGS.nb_heads,FLAGS.max_seq_length*2).cuda()

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
        if (reduce(operator.mul,k.shape,1) == 1368) and (d_batch * FLAGS.nb_heads * value_length * d_head_hidden == 24 * 19 * 9):
            print('stop')
        # This reshaping slices the last dimension and stacks those slices along the first dimension
        # In the resulting first dimension, first come all slices from the first batch, then from the second, and so on
        # This is relevant for how to add the position embeddings: they are the same per batch, but not per slice
        q_multi_parts = q.contiguous().view(d_batch * FLAGS.nb_heads, query_length, d_head_hidden)
        k_multi_parts = k.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)
        v_multi_parts = v.contiguous().view(d_batch * FLAGS.nb_heads, value_length, d_head_hidden)

        att_weights = torch.bmm(q_multi_parts, k_multi_parts.transpose(1, 2)) # shape [d_batch x nb_heads, query_length, value_length]

        if self.use_causal_mask:
            causal_mask = torch.tensor([[[1 if value_index <= query_index else 0 for value_index in range(att_weights.shape[2])] for query_index in range(att_weights.shape[1])] for _ in range(att_weights.shape[0])]).cuda()
            att_weights *= causal_mask

        att_weights = nn.Dropout()(att_weights)
        batch_pos_embeddings = self.select_pos_embeddings(query_length, value_length, d_batch)
        att_output_multi_parts = torch.bmm(att_weights + batch_pos_embeddings, v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(d_batch, query_length, FLAGS.d_hidden)
        return att_output

    def select_pos_embeddings(self, query_length, value_length, d_batch):
        rel_pos_indices = torch.tensor(
            [[[q_idx - k_idx for k_idx in range(value_length)] for q_idx in range(query_length)] for _ in
             range(FLAGS.nb_heads)]).cuda() # shape [nb_heads, query_length, value_length]
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
        return self.embedding_to_hidden(self.embedding_matrix[input].cuda())


def main(_):
    reader = GutenbergReader() #TODO add COPA task later
    train_dataset = reader.read(os.path.join(FLAGS.data_folder,'train'))
    test_dataset = reader.read(os.path.join(FLAGS.data_folder,'test'))
    val_dataset = reader.read(os.path.join(FLAGS.data_folder,'val'))

    vocab = Vocabulary.from_instances(train_dataset + val_dataset)
    vocab.add_token_to_namespace("@@PADDING@@")

    model = FullModel(vocab)
    cuda_device = 0
    model = model.cuda(cuda_device)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=FLAGS.d_batch, sorting_keys=[("inputs", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=FLAGS.patience,
                      num_epochs=FLAGS.num_epochs,
                      cuda_device=cuda_device)
    trainer.train() #TODO maybe add L2 penalty to loss


if __name__ == '__main__':
    app.run(main)
