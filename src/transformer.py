import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from collections import OrderedDict


class MultiSequential(nn.Sequential):
    """Sequential model with multiple inputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def seq_clones(module, depth):
    """Produce N identical layers."""

    return MultiSequential(
        OrderedDict(
            [('layer{}'.format(i), deepcopy(module)) for i in range(depth)]
        )
    )


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, nheads, masked=False):
        super().__init__()

        self.masked = masked
        self.nheads = nheads
        self.model_dim = model_dim

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        assert self.model_dim % self.nheads == 0

        key_dim = self.model_dim//self.nheads
        shape_q = query.shape[:2]+(self.nheads, key_dim)
        shape_k = key.shape[:2]+(self.nheads, key_dim)
        shape_v = value.shape[:2]+(self.nheads, key_dim)

        ret, att, = self.attention(self.linear_q(query).reshape(shape_q),
                                   self.linear_k(key).reshape(shape_k),
                                   self.linear_v(value).reshape(shape_v))
        ret = ret.reshape(ret.shape[:2] + (self.model_dim,))

        return self.linear_out(ret)

    def attention(self, query, key, value):
        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
        if self.masked:
            mask = torch.triu(torch.ones(score.shape, dtype=torch.bool),
                              diagonal=1)
            score[mask] = -float('inf')

        att = F.softmax(score / np.sqrt(score.shape[-1]), dim=-1)
        ret = torch.einsum('bhqk,bkhd->bqhd', att, value)

        return ret, att


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.encoder = nn.Linear(vocab_size, model_dim, bias=False)
        self.decoder = nn.Linear(model_dim, vocab_size, bias=False)

        self.decoder.weight = nn.Parameter(self.encoder.weight.t())

    def forward(self, x):
        if x.shape[-1] == self.model_dim:
            return self.decoder(x)

        return self.encoder(x) * np.sqrt(self.model_dim)


class PositionalEncoder(nn.Module):
    """Implement the PE function."""
    "Based on https://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, model_dim, hidden_dim, nheads):
        super().__init__()
        self.mhatt = MultiHeadAttention(model_dim, nheads)
        self.linear = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, src):
        src = self.mhatt(src, src, src)+src
        src = self.linear(src)+src

        return src


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, hidden_dim, nheads):
        super().__init__()
        self.mhatt_masked = MultiHeadAttention(
            model_dim, nheads, masked=True
        )
        self.mhatt = MultiHeadAttention(model_dim, nheads)

        self.linear = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, tgt, enc):
        tgt = self.mhatt_masked(tgt, tgt, tgt)+tgt
        tgt = self.mhatt(tgt, enc, enc)+tgt
        tgt = self.linear(tgt)+tgt

        return tgt, enc


class Transformer(nn.Module):

    def __init__(
            self,
            vocab_size,
            model_dim,
            hidden_dim,
            nheads,
            max_len=5000,
            depth=5
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, model_dim)
        self.pe = PositionalEncoder(model_dim, max_len)

        self.encoder = seq_clones(
            EncoderLayer(model_dim, hidden_dim, nheads), depth
        )
        self.decoder = seq_clones(
            DecoderLayer(model_dim, hidden_dim, nheads), depth
        )

        self.apply(self._init_weights)

    def forward(self, src, tgt):
        src = self.pe(self.embedding(src))
        tgt = self.pe(self.embedding(tgt))

        dec, _ = self.decoder(tgt, self.encoder(src))
        out = self.embedding(dec)

        return out

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
