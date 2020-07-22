import torch
from src.transformer.trainer import Trainer
from src.transformer.configuration_transformer import Config
from src.transformer.dataset import TextDataset
from src.transformer.transformer import Transformer
from torch.utils.tensorboard import SummaryWriter


def main():
    flags = Config(
        nheads=8,
        model_dim=512,
        hidden_dim=2048,
        depth=6,
        epochs=10,
        train_batch_size=32,
        eval_batch_size=32,
    )

    torch.manual_seed(flags.random_seed)
    torch.cuda.manual_seed(flags.random_seed)

    train_dataset = TextDataset(
        path_root='../../ml-datasets/wmt14/',
        path_src="train.en",
        path_tgt="train.de",
        path_tokenizer='tokenizer/',
    )

    eval_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
        )

    vocab_size = train_dataset.tokenizer.get_vocab_size()
    max_len = max(train_dataset.max_len, eval_dataset.max_len)
    model = Transformer(
        vocab_size=vocab_size,
        model_dim=flags.model_dim,
        hidden_dim=flags.hidden_dim,
        nheads=flags.nheads,
        max_len=max_len,
        depth=flags.depth,
    )

    train_op = Trainer(
        flags=flags,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tb_writer=SummaryWriter(),
        vocab_size=vocab_size,
    )
    train_op.fit()


if __name__ == "__main__":
    main()
