import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.model = model.to(device)
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
        print("Train on {0} samples, validate on {1} samples".format(
            self.train_dataset.len, self.eval_dataset.len
        ))

        for epoch in range(self.flags.epochs):
            print("Epoch {0}/{1}".format(epoch, self.flags.epochs))

            for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                batch_src, batch_tgt = batch
                batch_src = batch_src.to(device)
                batch_tgt = batch_tgt.to(device)

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
                self.tb_writer.add_scalar('Train/learning_rate', self._get_lr(), t)

                #sys.stdout.write("[%-60s] %d%%" % ('=' * (60 * (batch_idx + 1) // 10), (100 * (batch_idx + 1) // 10)))
                #sys.stdout.flush()
                #sys.stdout.write(", batch %d" % (batch_idx + 1))
                #sys.stdout.flush()

                if (batch_idx + 1) % self.flags.eval_rate == 0:
                    valid_loss = self.evaluate()
                    self.tb_writer.add_scalar('Valid/loss', valid_loss, t)

    def predict(self, inputs):
        with torch.no_grad():
            self.model.eval()

            outputs_dummy = torch.zeros(inputs.shape+(self.vocab_size,))
            return self._predict_loop(inputs.to(device), outputs_dummy.to(device))

    def evaluate(self):
        valid_loss = 0

        with torch.no_grad():
            self.model.eval()

            for batch_src, batch_tgt in tqdm(self.eval_dataloader):
                batch_src = batch_src.to(device)
                batch_tgt = batch_src.to(device)
                batch_dummy = torch.zeros(
                    batch_tgt.shape+(self.vocab_size,)
                ).to(device)
                outputs = self._predict_loop(batch_src, batch_dummy)

                print(outputs.is_cuda())

                valid_loss += self.loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    batch_tgt.reshape(-1)
                )

        num_batches = (len(self.eval_dataset)//self.flags.batch_size)
        return valid_loss/num_batches

    def _predict_loop(self, batch_src, batch_dummy):
        for _ in range(batch_dummy.shape[1]):
            batch_dummy = self.model(
                batch_src,
                torch.argmax(batch_dummy, dim=2)
            )

        return batch_dummy

    def _get_bleu_score(self, outputs, batch_tgt):
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

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
