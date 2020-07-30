class Statistics:
    def __init__(self, best_params, mean, minimum, std):
        self.min = minimum
        self.mean = mean
        self.std = std
        self.best_params = best_params

    @property
    def significance(self):
        return (self.mean - self.min) / self.std
