import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""
r  reset_gate
z  update_gate
h  activation
h~ candidate

reset_gate = sigmoid(mm(Wr,input)+mm(Ur,activation)+br)
update_gate = sigmoid(mm(Wz,input)+mm(Uz,activation)+bz)
candidate = tanh(mm(Wh,input)+mm(Uh,reset_gate * activation)+bh)
activation = (1-update_gate)*activation + update_gate * candidate
"""


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dp_keep_prob = 1.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(1 - dp_keep_prob)


        self.Wr = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.Ur = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.Wz = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.Uz = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.Wh = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.Uh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x, activation):
        x = self.dropout(x)
        reset_gate = torch.sigmoid(self.Wr(x) + self.Ur(activation))
        update_gate = torch.sigmoid(self.Wz(x) + self.Uz(activation))
        candidate = torch.tanh(self.Wh(x) + self.Uh(reset_gate * activation))

        activation = (1 - update_gate) * activation + update_gate * candidate
        return activation



# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN

  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):

    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(GRU, self).__init__()

    # TODO ========================

  def init_weights_uniform(self):
    # TODO ========================

  def init_hidden(self):
    # TODO ========================
    return # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    return samples