from typing import Optional
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
import pandas as pd


def pad_collate(batch):
    (x, y) = zip(*batch)

    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    y_pad = pad_sequence(y, batch_first=True, padding_value=0)

    return x_pad, y_pad


class TextDataset(Dataset):
    def __init__(
            self,
            path_src,
            path_tgt,
            path_tokenizer,
            path_root: Optional[str] = '',
    ):
        self.path_src = path_root+path_src
        self.path_tgt = path_root+path_tgt
        self.len = self._get_file_len()
        self.max_len = 512

        self.tokenizer = Tokenizer(BPE(
            path_root + path_tokenizer + 'vocab.json',
            path_root + path_tokenizer + 'merges.txt',
        ))
        self.tokenizer.normalizer = Sequence([
            NFKC(),
            Lowercase()
        ])

    def _encode(self, src_line, tgt_line):
        if len(src_line) > self.max_len:
            self.max_len = len(src_line)

        if len(tgt_line) > self.max_len:
            self.max_len = len(tgt_line)

        src = self.tokenizer.encode(str(src_line)).ids
        tgt = self.tokenizer.encode(str(tgt_line)).ids

        return torch.tensor(src), torch.tensor(tgt)

    def __len__(self):
        return self.len

    def _get_file_len(self):
        with open(self.path_src, "r") as f:
            return sum(bl.count("\n") for bl in self._blocks(f))

    @staticmethod
    def _blocks(files, size=65536):
        b = ' '
        while b:
            b = files.read(size)
            yield b

    def __getitem__(self, i):
        reader_src = pd.read_csv(
            self.path_src, sep='\n', skiprows=i, iterator=True,
        )
        reader_tgt = pd.read_csv(
            self.path_tgt, sep='\n', skiprows=i, iterator=True,
        )

        return self._encode(
            reader_src.get_chunk(1),
            reader_tgt.get_chunk(1),
        )
