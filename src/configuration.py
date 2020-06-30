class Config:

    def __init__(
            self,
            random_seed=1234,
            nheads=8,
            key_dim=64,
            model_dim=512,
            hidden_size=128,
            depth=5,
            max_len=5000,
            batch_size=64,
            lr=3e-4
    ):

        self.random_seed = random_seed
        self.nheads = nheads
        self.key_dim = key_dim
        self.model_dim = model_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
