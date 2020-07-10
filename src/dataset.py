from typing import Optional
from torch.utils.data import Dataset
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence


def pad_collate(batch):
    (xx, yy) = zip(*batch)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad


class TextDataset(Dataset):
    def __init__(
            self,
            path_src,
            path_tgt,
            path_tokenizer,
            path_root: Optional[str] = '',
    ):
        self.max_len = 0

        self.tokenizer = Tokenizer(BPE(
            path_root + path_tokenizer + 'vocab.json',
            path_root + path_tokenizer + 'merges.txt',
        ))
        self.tokenizer.normalizer = Sequence([
            NFKC(),
            Lowercase()
        ])

        self.examples = []

        src_file = Path(path_root+path_src)
        tgt_file = Path(path_root + path_tgt)
        src_lines = src_file.read_text(encoding="utf-8").splitlines()
        tgt_lines = tgt_file.read_text(encoding="utf-8").splitlines()

        self.examples = list(map(self._encode, src_lines, tgt_lines))

    def _encode(self, src_line, tgt_line):
        if len(src_line) > self.max_len:
            self.max_len = len(src_line)

        if len(tgt_line) > self.max_len:
            self.max_len = len(tgt_line)

        src = self.tokenizer.encode(src_line).ids
        tgt = self.tokenizer.encode(tgt_line).ids

        return torch.tensor(src), torch.tensor(tgt)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
