class Trainer:

    def __init__(
            self,
            model,
            train_dataset,
            eval_dataset,
            tb_writer,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tb_writer = tb_writer

    def _get_optimizer(self):
        pass

    def _get_loss_fn(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def _predict_loop(self):
        pass

    def _get_bleu_score(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
