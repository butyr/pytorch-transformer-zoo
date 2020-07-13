# pytorch-transformer
Pytorch implementation of "Attention is all you need"

## Installation

Install via pip:

```
pip install git+https://github.com/butyr/pytorch-transformer.git
```

## Example
```python
from transformer.trainer import *
from transformer.configurations import *
from transformer.dataset import *
from transformer.transformer import *
from torch.utils.tensorboard import SummaryWriter


def main():
    flags = Config(
        nheads=8,
        model_dim=512,
        hidden_dim=2048,
        depth=5,
        epochs=10,
    )

    train_dataset = TextDataset(
        path_root='../../ml-datasets/wmt14/',
        path_src="newstest2014.en",
        path_tgt="newstest2014.de",
        path_tokenizer='tokenizer/',
        right_shift=True,
    )

    eval_dataset = TextDataset(
            path_root='../../ml-datasets/wmt14/',
            path_src="newstest2014.en",
            path_tgt="newstest2014.de",
            path_tokenizer='tokenizer/',
            right_shift=True,
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

```