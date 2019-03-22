import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def masked_softmax(x, mask):
    if mask is not None:
        mask = mask.unsqueeze(1)
        x = x.masked_fill(mask == 0, -1e9)

    return F.softmax(x, dim=-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert n_units % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = n_units // n_heads
        self.n_heads = n_heads
        self.n_units = n_units

        # Linear layers for Q, K, V and output
        self.Wq, self.Wk, self.Wv, self.Wo = clones(nn.Linear(n_units, n_units), 4)
        self.dropout = nn.Dropout(dropout)

        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        self.init_parameters()

    def self_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scale = np.sqrt(d_k)

        QK = torch.matmul(Q, K.transpose(-2, -1)) / scale
        Ai = masked_softmax(QK, mask)
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
        Q = self.Wq(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        H = self.self_attention(Q, K, V, mask=mask)

        # concatenate all heads
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_units)

        A = self.Wo(H)

        return A  # size: (batch_size, seq_len, self.n_units)

    def init_parameters(self):
        k = 1.0 / np.sqrt(self.n_units)

        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            torch.nn.init.uniform_(layer.weight, -k, k)
            torch.nn.init.uniform_(layer.bias, -k, k)


#####################################################


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert n_units % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = n_units // n_heads
        self.h = n_heads
        self.n_units = n_units
        self.linears = clones(nn.Linear(n_units, n_units), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.init_parameters()

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def init_parameters(self):

        k = 1.0 / np.sqrt(self.n_units)

        for layer in self.linears:
            torch.nn.init.uniform_(layer.weight, -k, k)
            torch.nn.init.uniform_(layer.bias, -k, k)