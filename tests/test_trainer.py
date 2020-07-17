import unittest
import torch
from src.transformer.trainer import Trainer
from src.transformer.configurations import Config
from src.transformer.dataset import TextDataset
from src.transformer.transformer import Transformer


class TestTrainer(unittest.TestCase):
  
    def setUp(self):
        flags = Config(
            nheads=2,
            model_dim=10,
            hidden_dim=10,
            depth=2,
            epochs=1,
            train_batch_size=64,
        )

        train_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
        )

        eval_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
        )

        self.vocab_size = train_dataset.tokenizer.get_vocab_size()
        max_len = max(train_dataset.max_len, eval_dataset.max_len)
        model = Transformer(
            vocab_size=self.vocab_size,
            model_dim=flags.model_dim,
            hidden_dim=flags.hidden_dim,
            nheads=flags.nheads,
            max_len=max_len,
            depth=flags.depth,
        )

        self.train_op = Trainer(
            flags=flags,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tb_writer=None,
            vocab_size=self.vocab_size,
        )
        self.tokenizer = self.train_op.train_dataset.tokenizer

    def test_bleu_score(self):
        batch_src, batch_tgt = next(iter(self.train_op.train_dataloader))
        one_hot = self.to_one_hot(batch_tgt, self.vocab_size)
        bleu = self.train_op._get_bleu_score(one_hot, batch_tgt)

        self.assertEqual(1.0, bleu)

    @staticmethod
    def to_one_hot(y, n_dims=None):
        y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot
