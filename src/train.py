from src.transformer import *
from src.configuration import Config
from torch.utils.tensorboard import SummaryWriter


config = Config()
writer = SummaryWriter()
torch.manual_seed(config.random_seed)

vocab_size = 10  # write dataset to get vocabsize
max_len = 10  # write dataset to get max_len
train_data = [([], [])]
val_data = [([], [])]

model = Transformer(
    vocab_size,
    config.model_dim,
    config.hidden_dim,
    config.nheads,
    max_len,
    config.depth
)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for src, tgt in train_data:
    outputs = model(src, tgt)
    loss = loss_fn(
        outputs.reshape(-1, config.model_dim), tgt.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""
# add to the end of training
writer.add_hparams({
    'lr': config.lr,
    'bsize': config.batch_size,
    'd_k': config.key_dim,
    'h': config.nheads,
    'n': config.depth,
    'hsize': config.hidden_dim
}, {
    'hparam/bleu': 0,
    'hparam/loss': 0
})
"""
