import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Trainer:

    def __init__(
            self,
            flags,
            model,
            train_dataset,
            eval_dataset,
            tb_writer,
            vocab_size,
            save_path=None,
    ):
        self.flags = flags
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = self._get_dataloader(train=True)
        self.eval_dataloader = self._get_dataloader()
        self.tb_writer = tb_writer
        self.vocab_size = vocab_size
        self.save_path = save_path if save_path is not None else '../checkpoints/model.ckp'

        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss_fn()

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), self.flags.lr)

    @staticmethod
    def _get_loss_fn():
        return nn.CrossEntropyLoss()

    def fit(self):
        for epoch in range(self.flags.epochs):
            for batch_idx, batch in enumerate(self.train_dataloader):
                batch_src, batch_tgt = batch
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(batch_src, batch_tgt)
                loss = self.loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    batch_tgt.reshape(-1)
                )
                loss.backward()
                self.optimizer.step()

                t = (epoch * len(batch)) + batch_idx
                self.tb_writer.add_scalar('Train/loss', loss, t)
                self.tb_writer.add_scalar(
                    'Train/bleu', self._get_bleu_score(outputs, batch_tgt), t
                )
                self.tb_writer.add_scalar('Train/learning_rate', self._get_lr(), t)

                if (batch_idx + 1) % self.flags.eval_rate == 0:
                    valid_loss, bleu = self.evaluate()
                    self.tb_writer.add_scalar('Valid/loss', valid_loss, t)
                    self.tb_writer.add_scalar('Train/bleu', bleu, t)

    def predict(self, inputs):
        with torch.no_grad():
            self.model.eval()

            outputs_dummy = torch.zeros(inputs.shape+(self.vocab_size,))
            return self._predict_loop(inputs, outputs_dummy)

    def evaluate(self):
        valid_loss = 0
        bleu = 0

        with torch.no_grad():
            self.model.eval()

            for batch_src, batch_tgt in self.eval_dataloader:
                batch_dummy = torch.zeros(batch_src.shape+(self.vocab_size,))
                outputs = self._predict_loop(batch_src, batch_dummy)

                valid_loss += self.loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    batch_tgt.reshape(-1)
                )
                bleu += self._get_bleu_score(outputs, batch_tgt)

        num_batches = (len(self.eval_dataset)//self.flags.batch_size)
        return valid_loss/num_batches, bleu/num_batches

    def _predict_loop(self, batch_src, batch_dummy):
        for _ in range(batch_dummy.shape[1]):
            batch_dummy = self.model(
                batch_src,
                torch.argmax(batch_dummy, dim=-1)
            )

        return batch_dummy

    def _get_bleu_score(self, outputs, batch_tgt):
        candidate_corpus = self.train_dataset.tokenizer.decode(
            torch.argmax(outputs, dim=-1)
        )
        references_corpus = self.train_dataset.tokenizer.decode(batch_tgt)

        return bleu_score(candidate_corpus, references_corpus)

    def save_model(self):
        pass

    def load_model(self):
        pass

    def _get_dataloader(self, train=False):

        if train:
            return DataLoader(
                self.train_dataset,
                batch_size=self.flags.batch_size,
                shuffle=self.flags.train_shuffle,
                num_workers=self.flags.num_workers,
                collate_fn=self.train_dataset.pad_collate,
            )
        else:
            return DataLoader(
                self.eval_dataset,
                batch_size=self.flags.batch_size,
                num_workers=self.flags.num_workers,
                collate_fn=self.eval_dataset.pad_collate,
            )

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
