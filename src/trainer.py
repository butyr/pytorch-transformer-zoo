class Trainer:

    def __init__(self, flags):
        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss_fn()

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
