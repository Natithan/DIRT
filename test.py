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

matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import torch.optim as optim

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
flags.DEFINE_integer("source_length", 20, "Number of tokens in source sequence")
flags.DEFINE_integer("max_seq_length", 20, "Maximum number of words to consider per batch")
flags.DEFINE_string("data_folder", "./data/Gutenberg", "Folder with train, val and test subfolders containing data")

# Trainer flags
flags.DEFINE_integer("patience", 10, "Number of epochs the validation metric can worsen before stopping training.")
flags.DEFINE_integer("num_epochs", 1000, "Numnber of epochs to train for.")

flags.DEFINE_bool("mini", True, "Whether to work with mini data/models for debugging purposes")

# %%

class GutenbergReader(DatasetReader):

    def __init__(self, token_indexers=None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens, tags=None):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

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
    def __init__(self):
        self.attention = AttentionLayer()

    def forward(self, input):
        embedded = self.embedding(input)
        encoded = self.attention(embedded)
        # TODO decide on task: MLM, LM, permutation LM





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



class MyModel(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embedder = AlbertEmbedder(vocab)

    def forward(self, *input: Any, **kwargs: Any) -> T_co:



def main(_):
    reader = GutenbergReader()
    train_dataset = reader.read(os.path.join(FLAGS.data_folder,'train'))
    test_dataset = reader.read(os.path.join(FLAGS.data_folder,'test'))
    val_dataset = reader.read(os.path.join(FLAGS.data_folder,'val'))
    vocab = Vocabulary.from_instances(train_dataset + val_dataset)

    model = MyModel(vocab)
    cuda_device = 0
    model = model.cuda(cuda_device)

    optimizer = optim.Adam()

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
