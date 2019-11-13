import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import nltk
from torch.nn.modules.activation import MultiheadAttention
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("d_batch", 3, "Batch size")
flags.DEFINE_integer("d_emb", 16, "Embedding size")
flags.DEFINE_integer("nb_heads", 8, "Number of attention heads")
flags.DEFINE_integer("target_length", 2, "Number of tokens in target sequence")
flags.DEFINE_integer("source_length", 2, "Number of tokens in source sequence")

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiHeadAttention()
        self.feedforward = nn.Linear(FLAGS.d_emb,FLAGS.d_emb)

    def forward(self, input):
        att_out = self.multihead_attention(input) + input # Include skip-connection
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
        q_multi_parts = q.contiguous().view(FLAGS.d_batch*FLAGS.nb_heads,FLAGS.target_length,d_head_emb)
        k_multi_parts = k.contiguous().view(FLAGS.d_batch*FLAGS.nb_heads,FLAGS.source_length,d_head_emb)
        v_multi_parts = v.contiguous().view(FLAGS.d_batch*FLAGS.nb_heads,FLAGS.source_length,d_head_emb)
        att_weights = torch.bmm(q_multi_parts,k_multi_parts.transpose(1,2))
        att_output_multi_parts = torch.bmm(att_weights,v_multi_parts)
        att_output = att_output_multi_parts.contiguous().view(FLAGS.d_batch,FLAGS.target_length,FLAGS.d_emb)
        return att_output

def main(_):
    input = torch.rand(FLAGS.d_batch,FLAGS.source_length,FLAGS.d_emb)
    output = AttentionLayer()(input)
    print(output)

if __name__ == '__main__':
  app.run(main)