import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import pad_collate


class Trainer:

    def __init__(
            self,
            flags,
            model,
            train_dataset,
            eval_dataset,
            tb_writer,
    ):
        self.flags = flags
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer

        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss_fn()

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), self.flags.lr)

    @staticmethod
    def _get_loss_fn():
        return nn.CrossEntropyLoss()

    def fit(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.flags.batch_size,
            shuffle=self.flags.train_shuffle,
            num_workers=self.flags.num_workers,
            collate_fn=pad_collate,
        )

        self.model.train()

        for batch_src, batch_tgt in dataloader:
            print(batch_src.shape)
            print(batch_tgt.shape)
            self.model(batch_src, batch_tgt)
            break

    def predict(self):
        pass

    def evaluate(self):
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.flags.batch_size,
            num_workers=self.flags.num_workers,
            collate_fn=pad_collate,
        )

        self.model.eval()

        for batch_src, batch_tgt in dataloader:
            break

    def _predict_loop(self):
        pass

    def _get_bleu_score(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
