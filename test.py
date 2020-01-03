# %% Imports
from __future__ import unicode_literals, print_function

import os
from pathlib import Path
from typing import Any, T_co

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

import nltk
from torch.nn.modules.activation import MultiheadAttention
from absl import app
from absl import flags

# %% FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 3, "Batch size")
flags.DEFINE_integer("d_emb", 12, "Size of token encodings before contextualization")
flags.DEFINE_integer("d_hidden", 72, "Size of token encodings in hidden layers (contextualized)")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
flags.DEFINE_integer("target_length", 20, "Number of tokens in target sequence")
flags.DEFINE_float("masking_fraction", .15, "Fraction of tokens to be masked during MLM pretraining")
flags.DEFINE_integer("source_length", 20, "Number of tokens in source sequence")
flags.DEFINE_integer("max_seq_length", 20, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", "./data/Gutenberg", "Folder with train, val and test subfolders containing data")

# Trainer flags
flags.DEFINE_integer("patience", 10, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 1000, "Number of epochs to train for.")

flags.DEFINE_bool("mini", True, "Whether to work with mini data/models for debugging purposes")

# %%

def t5_denoise_spans_objective(tokens): # Based on objective in t5 paper: https://arxiv.org/abs/1910.10683
    masked_indices = sorted(random.sample(range(len(tokens)),int(len(tokens)*FLAGS.masking_fraction))) # TODO finish this creating of given and target fields

    given = [t if (i not in masked_indices) else '@@MASK@@' for i, t in enumerate(tokens)]
    masked_given = [i for i, j in zip(given[1:], given[:-1]) if not (i == '@@MASK@@' and i == j)]
    mask_counter = itertools.count()
    unique_masked_given = [Token(f'{i}_{next(mask_counter)}') if i == '@@MASK@@' else i for i in masked_given]

    target = [tokens[i] for i in masked_indices]
    include_mask = [True] + [((i - j) != 1) for i, j in zip(masked_indices[1:], masked_indices[:-1])]
    masks = ['@@MASK@@' if x else '@@TO_BE_DELETED@@' for x in include_mask]
    masked_target = [i for j in zip(masks, target) for i in j if i != '@@TO_BE_DELETED@@']
    mask_counter = itertools.count() #Restart count
    unique_masked_target = [Token(f'{i}_{next(mask_counter)}') if i == '@@MASK@@' else i for i in masked_target]
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

class FullModel(nn.Module):
    def __init__(self,vocab):
        """
        phase: either 'pretrain' or
        """

        self.embedder = AlbertEmbedder(vocab)
        self.attention = AttentionLayer()

    def forward(self, input):
        embedded = self.embedder(input)
        encoded = self.attention(embedded)
        prediction = self.predict(encoded,self.phase)
        # TODO decide on task: MLM, LM, permutation LM, Ernies masking
        # TODO add text-to-text objective like t5?


    def lm(dataset): # inspired by from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
        """Basic language modeling objective for text - empty inputs.
        Given inputs with the format:
        {"text": "Here is some text."}
        This preprocess produces examples with the format
        {"inputs": "", "targets": "Here is some text."}
        Args:
          dataset: A tf.data.Dataset to process.
        Returns:
          A preprocessed tf.data.Dataset.
        """
        return dataset.map(
            lambda x: {'inputs': '', 'targets': x['text']},
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )



class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiHeadAttention()
        self.feedforward = nn.Linear(FLAGS.d_emb, FLAGS.d_emb)

    def forward(self, input):
        att_out = self.multihead_attention(input) + input  # Include skip-connection
        ff_out = self.feedforward(att_out) + att_out
        return ff_out


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.project_q = nn.Linear(FLAGS.d_emb, FLAGS.d_emb)
        self.project_k = nn.Linear(FLAGS.d_emb, FLAGS.d_emb)
        self.project_v = nn.Linear(FLAGS.d_emb, FLAGS.d_emb)

    def forward(self, input):
        q = self.project_q(input)
        k = self.project_k(input)
        v = self.project_v(input)
        assert FLAGS.d_emb % FLAGS.nb_heads == 0
        d_head_emb = FLAGS.d_emb // FLAGS.nb_heads
        q_multi_parts = q.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, FLAGS.target_length, d_head_emb)
        k_multi_parts = k.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, FLAGS.source_length, d_head_emb)
        v_multi_parts = v.contiguous().view(FLAGS.d_batch * FLAGS.nb_heads, FLAGS.source_length, d_head_emb)
        att_weights = torch.bmm(q_multi_parts, k_multi_parts.transpose(1, 2))
        att_output_multi_parts = torch.bmm(att_weights, v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(FLAGS.d_batch, FLAGS.target_length, FLAGS.d_emb)
        return att_output


class AlbertEmbedder(nn.Module):
    """
    Factorized embedder, as proposed in the ALBERT paper: http://arxiv.org/abs/1909.11942
    """
    def __init__(self,vocab):
        super().__init__()
        self.index_to_embedding = nn.Linear(vocab.get_vocab_size('tokens'), FLAGS.d_emb)
        self.embedding_to_hidden = nn.Linear(FLAGS.d_emb, FLAGS.d_hidden)
    def forward(self, input):
        return self.embedding_to_hidden(self.index_to_embedding(input))


def main(_):
    reader = GutenbergReader() #TODO add COPA task later
    train_dataset = reader.read(os.path.join(FLAGS.data_folder,'train'))
    test_dataset = reader.read(os.path.join(FLAGS.data_folder,'test'))
    val_dataset = reader.read(os.path.join(FLAGS.data_folder,'val'))

    # masking_tokens = [f'MASK_{i}' for i in range(int(FLAGS.target_length*FLAGS.masking_fraction))]
    # masking_token_instances = [reader.text_to_instance([Token(word) for word in masking_tokens])]

    vocab = Vocabulary.from_instances(masking_token_instances + train_dataset + val_dataset)

    model = FullModel(vocab)
    cuda_device = 0
    model = model.cuda(cuda_device)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=FLAGS.d_batch, sorting_keys=[("sentence", "num_tokens")])
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
    output = AttentionLayer()(input)
    # loss =
    print(output)


if __name__ == '__main__':
    app.run(main)
