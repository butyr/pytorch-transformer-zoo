from trainer import *
from configurations import *
from transformer import *


def main(_):
    flags = Config()
    train_op = trainer.Trainer(flags)
    train_op.fit()


if __name__ == "__main__":
    main()
