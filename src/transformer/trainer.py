import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(
            self,
            flags,
            model,
            train_dataset,
            eval_dataset,
            tb_writer,
            vocab_size,
    ):
        self.flags = flags
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = self._get_dataloader(train=True)
        self.eval_dataloader = self._get_dataloader()
        self.tb_writer = tb_writer
        self.vocab_size = vocab_size

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

                if (batch_idx + 1) % self.flags.eval_rate == 0:
                    self.evaluate()

    def predict(self, inputs):
        with torch.no_grad():
            self.model.eval()

            outputs_dummy = torch.zeros_like(inputs, dtype=torch.long)
            return self._predict_loop(
                inputs.to(device),
                outputs_dummy.to(device)
            )

    def evaluate(self):
        valid_loss = 0

        with torch.no_grad():
            self.model.eval()

            for batch_src, batch_tgt in self.eval_dataloader:
                batch_dummy = torch.zeros_like(
                    batch_tgt.to(device),
                    dtype=torch.long
                )
                outputs = self._predict_loop(
                    batch_src.to(device),
                    batch_dummy.to(device)
                )

                valid_loss += self.loss_fn(outputs, batch_tgt)

    def _predict_loop(self, batch_src, batch_dummy):
        for _ in range(batch_dummy.shape[-1]):
            batch_dummy = torch.LongTensor(self.model(
                batch_src.to(device),
                batch_dummy.to(device)
            ))

        return batch_dummy

    def _get_bleu_score(self):
        pass

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
