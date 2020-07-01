import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from collections import OrderedDict


class MultiSequential(nn.Sequential):
    """Sequential model with as many inputs as outputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def seq_clones(module, n):
    """Produce n identical layers."""

    return MultiSequential(
        OrderedDict(
            [('layer{}'.format(i), deepcopy(module))
             for i in range(n)]
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

        ret, att, = self.attention(
            self.linear_q(query).reshape(shape_q),
            self.linear_k(key).reshape(shape_k),
            self.linear_v(value).reshape(shape_v)
        )
        ret = ret.reshape(ret.shape[:2] + (self.model_dim,))

        return self.linear_out(ret)

    def attention(self, query, key, value):
        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
        if self.masked:
            mask = torch.triu(
                torch.ones(score.shape, dtype=torch.bool), diagonal=1
            )
            score[mask] = -float('inf')

        att = F.softmax(score / np.sqrt(score.shape[-1]), dim=-1)
        ret = torch.einsum('bhqk,bkhd->bqhd', att, value)

        return ret, att


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.encoder = nn.Embedding(vocab_size, model_dim)
        self.decoder = nn.Linear(model_dim, vocab_size, bias=False)

        self.decoder.weight = self.encoder.weight

    def forward(self, x, inverse=False):
        if inverse:
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
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, src):
        src_att = self.mhatt(src, src, src)+src
        src_out = self.ffn(src_att)+src_att

        return src_out


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, hidden_dim, nheads):
        super().__init__()
        self.mhatt_masked = MultiHeadAttention(
            model_dim, nheads, masked=True
        )
        self.mhatt = MultiHeadAttention(model_dim, nheads)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, tgt, enc):
        tgt_att1 = self.mhatt_masked(tgt, tgt, tgt)+tgt
        tgt_att2 = self.mhatt(tgt_att1, enc, enc)+tgt_att1
        tgt_out = self.ffn(tgt_att2)+tgt_att2

        return tgt_out, enc


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

        self.src_embedding = None
        self.tgt_embedding = None

    def forward(self, src, tgt):
        self.src_embedding = self.embedding(src)
        self.tgt_embedding = self.embedding(tgt)

        src_pe = self.pe(self.src_embedding)
        tgt_pe = self.pe(self.tgt_embedding)

        dec, _ = self.decoder(tgt_pe, self.encoder(src_pe))

        return self.embedding(dec, inverse=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
