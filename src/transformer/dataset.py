from typing import Optional
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence


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
        self.len = 0
        self.max_len = 512

        self.tokenizer = Tokenizer(
            BPE(
                path_root + path_tokenizer + 'vocab.json',
                path_root + path_tokenizer + 'merges.txt',
            )
        )
        self.tokenizer.normalizer = Sequence([
            NFKC(),
            Lowercase()
        ])

        with open(self.path_src, 'r+') as f:
            lines_src = f.readlines()

        with open(self.path_tgt, 'r+') as f:
            lines_tgt = f.readlines()

        self.len = len(lines_src)
        self.example = list(zip(lines_src, lines_tgt))

    def _encode(self, src_line, tgt_line):
        src = self.tokenizer.encode(str(src_line)).ids
        tgt = self.tokenizer.encode(str(tgt_line)).ids

        if len(src) > self.max_len:
            self.max_len = len(src)

        if len(tgt) > self.max_len:
            self.max_len = len(tgt)

        return torch.tensor(src), torch.tensor(tgt), len(src), len(tgt)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self._encode(*self.example[i])

    @staticmethod
    def pad_collate(batch):
        (x, y, x_len, y_len) = zip(*batch)

        x_pad = pad_sequence(x, batch_first=True, padding_value=0)
        y_pad = pad_sequence(y, batch_first=True, padding_value=0)

        return x_pad, y_pad, x_len, y_len
