# pytorch-transformer-zoo

This repository aims at providing the main variations of the transformer model in PyTorch. 
Currently it includes the initial model based on "Attention Is All You Need" 
([Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)) and the OpenAI GPT2 model based on 
[Radford et al. 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
and [Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).


## Installation

Install via pip:

```
pip install git+https://github.com/butyr/pytorch-transformer.git
```

## Usage

```python
from transformer import Transformer
import torch

model = Transformer(
        vocab_size=25_000,
        model_dim=512,
        hidden_dim=2048,
        nheads=8,
        max_len=512,
        depth=6,
    )

src_batch = # Tensor with shape (batch_size, src_sentence_length)
tgt_batch = # Tensor with shape (batch_size, tgt_sentence_length)

# outputs with teacher forcing
outputs = model(src_batch, tgt_batch)

# outputs without teacher forcing
dummy_batch = torch.zeros((batch_size, tgt_sentence_length, vocab_size))

for _ in range(tgt_sentence_length):
    dummy_batch = model(
        src_batch,
        torch.argmax(dummy_batch, dim=2)
    )
outputs = dummy_batch

```

## Example
```python
import torch
from transformer import Trainer
from transformer import TransformerConfig
from transformer.dataset import TextDataset
from transformer import Transformer
from torch.utils.tensorboard import SummaryWriter


def main():
    flags = TransformerConfig(
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


```
