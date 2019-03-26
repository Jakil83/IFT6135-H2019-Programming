import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def masked_softmax(x, mask):
    if mask is not None:
        # reshape on heads
        mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        x = x.masked_fill(mask == 0, -1e9)

    return F.softmax(x,dim=-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units

        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers for Q, K, V and output

        linear = nn.Linear(self.n_units, self.n_units)
        self.Wq, self.Wk, self.Wv, self.Wo = clones(linear, 4)

        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        self.init_parameters()

    def self_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scale = np.sqrt(d_k)

        QK = torch.matmul(Q, K.transpose(-2, -1)) / scale
        Ai = masked_softmax(QK, mask)
        if self.dropout is not None:
            Ai = self.dropout(Ai)
        Hi = torch.matmul(Ai, V)

        return Hi

    def forward(self, query, key, value, mask=None):

        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        batch_size = query.size(0)
        seq_len = query.size(1)

        # batch_size, seq_len, n_units --> batch_size, heads, seq_len, d_k
        Q = self.Wq(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        H = self.self_attention(Q, K, V, mask)

        # concatenate all heads
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_units)

        A = self.Wo(H)

        return A  # size: (batch_size, seq_len, self.n_units)

    def init_parameters(self):

        k = 1.0 / np.sqrt(self.n_units)

        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            layer.weight.data.uniform_(-k, k)
            layer.bias.data.uniform_(-k, k)