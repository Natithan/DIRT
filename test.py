# %% Imports
from __future__ import unicode_literals, print_function

import os
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
                if i > 5:
                    break
            with open(file) as f:
                running_sequence = []
                for line in f:
                    words = line.strip().split()
                    running_sequence += words
                    if len(running_sequence) >= FLAGS.max_seq_length:
                        current_sequence = running_sequence[:FLAGS.max_seq_length]
                        running_sequence = running_sequence[FLAGS.max_seq_length:]
                        yield self.text_to_instance([Token(word) for word in current_sequence])


def f(prediction, targets): #TODO
    pass


class FullModel(Model):
    def __init__(self,vocab):
        """
        """
        super().__init__(vocab)

        self.embedder = AlbertEmbedder(vocab)
        self.encoder = nn.Sequential(*[EncoderBlock() for _ in range(FLAGS.nb_encoder_layers)])
        self.decoder = MySequential(*[DecoderBlock() for _ in range(FLAGS.nb_encoder_layers)])


    def forward(self, inputs, targets):
        embedded_inputs = self.embedder(inputs['tokens'])
        embedded_outputs = torch.rand(FLAGS.d_batch, int(FLAGS.max_seq_length*FLAGS.masking_fraction*2 + 1),FLAGS.d_hidden).cuda() # Longest length if no adjacent masks
        encoded = self.encoder(nn.Dropout()(embedded_inputs))
        decoded,_ = self.decoder(nn.Dropout()(embedded_outputs),encoded)
        prediction = nn.Softmax()(nn.Linear(FLAGS.d_hidden, FLAGS.d_vocab)(nn.Dropout()(decoded))) #TODO add d_vocab
        loss = f(prediction,targets)
        return loss

        # TODO add text-to-text objective like t5? Probs by adding decoder
        # TODO add position embedding

    # def lm(dataset): # inspired by from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    #     """Basic language modeling objective for text - empty inputs.
    #     Given inputs with the format:
    #     {"text": "Here is some text."}
    #     This preprocess produces examples with the format
    #     {"inputs": "", "targets": "Here is some text."}
    #     Args:
    #       dataset: A tf.data.Dataset to process.
    #     Returns:
    #       A preprocessed tf.data.Dataset.
    #     """
    #     return dataset.map(
    #         lambda x: {'inputs': '', 'targets': x['text']},
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #     )


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


class DecoderBlock(nn.Module): # TODO
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

    def __init__(self,use_causal_mask=False):
        super().__init__()
        self.project_q = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_k = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.project_v = nn.Linear(FLAGS.d_hidden, FLAGS.d_hidden)
        self.use_causal_mask = use_causal_mask

    def forward(self, query,values):
        q = self.project_q(query)
        k = self.project_k(values)
        v = self.project_v(values)
        assert FLAGS.d_hidden % FLAGS.nb_heads == 0
        d_head_hidden = FLAGS.d_hidden // FLAGS.nb_heads
        query_length = query.shape[1]
        value_length = values.shape[1]
        q_multi_parts = q.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, query_length, d_head_hidden)
        k_multi_parts = k.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, value_length, d_head_hidden)
        v_multi_parts = v.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, value_length, d_head_hidden)
        att_weights = torch.bmm(q_multi_parts, k_multi_parts.transpose(1, 2))
        if self.use_causal_mask:
            causal_mask = torch.tensor([[[1 if value_index <= query_index else 0 for value_index in range(att_weights.shape[2])] for query_index in range(att_weights.shape[1])] for _ in range(att_weights.shape[0])]).cuda()
            att_weights *= causal_mask
        att_weights = nn.Dropout()(att_weights) #TODO add causal masking here
        att_output_multi_parts = torch.bmm(att_weights, v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(FLAGS.d_batch, query_length, FLAGS.d_hidden)
        return att_output


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
    trainer.train()
    input = torch.rand(FLAGS.d_batch, FLAGS.source_length, FLAGS.d_emb)
    output = EncoderBlock()(input)
    # loss =
    print(output)


if __name__ == '__main__':
    app.run(main)
