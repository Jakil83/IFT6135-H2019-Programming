import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy

np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
from models import make_model as TRANSFORMER

import matplotlib.pyplot as plt


torch.manual_seed(1111)


def normalize(array):

    array = np.array(array)

    return (array - array.min())/(array.max() - array.min())


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = int(len(raw_data)/100)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


def problem_5_1():


    return None



def problem_5_2():

    model_RNN = RNN(emb_size=200, hidden_size=1500,
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2,
                dp_keep_prob=0.35)

    model_GRU = GRU(emb_size=200, hidden_size=1500,
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2,
                dp_keep_prob=0.35)

    hidden_1 = model_RNN.init_hidden()
    hidden_1 = hidden_1.to(device)

    hidden_2 = model_GRU.init_hidden()
    hidden_2 = hidden_2.to(device)

    checkpoint = torch.load('best_models/RNN/best_params.pt', map_location=device)
    model_RNN.load_state_dict(checkpoint)

    checkpoint = torch.load('best_models/GRU/best_params.pt', map_location=device)
    model_GRU.load_state_dict(checkpoint)

    word_to_id, id_2_word = _build_vocab('data/ptb.valid.txt')
    valid_data = _file_to_word_ids('data/ptb.valid.txt', word_to_id)
    # Same batch for both models
    (x, y) = next(ptb_iterator(valid_data, model_RNN.batch_size, model_RNN.seq_len))

    inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)  # .cuda()

    model_GRU.zero_grad()
    model_RNN.zero_grad()

    hidden_1 = repackage_hidden(hidden_1)
    hidden_2 = repackage_hidden(hidden_2)

    outputs_RNN, hidden_RNN = model_RNN(inputs, hidden_1)
    outputs_GRU, hidden_GRU = model_GRU(inputs, hidden_2)

    targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)  # .cuda()


    norms_RNN = []
    norms_GRU = []

    loss = nn.CrossEntropyLoss(reduction='mean')(outputs_RNN.contiguous()[34, :], targets[34, :])
    for i in range(model_RNN.seq_len):
        concatenated = torch.cat(torch.autograd.grad(loss, model_RNN.all_hidden_states[i], retain_graph=True), dim=1)
        norm = torch.norm(concatenated, dim=1).mean()
        norms_RNN.append(norm.data.item())

    loss = nn.CrossEntropyLoss(reduction='mean')(outputs_GRU.contiguous()[34, :], targets[34, :])
    for i in range(model_GRU.seq_len):
        concatenated = torch.cat(torch.autograd.grad(loss, model_GRU.all_hidden_states[i], retain_graph=True), dim=1)
        norm = torch.norm(concatenated, dim=1).mean()
        norms_GRU.append(norm.data.item())


    plt.plot(normalize(norms_RNN), label='RNN')
    plt.plot(normalize(norms_GRU), label='GRU')
    plt.legend(loc='upper left')
    plt.show()


def problem_5_3():

    return #None samples
