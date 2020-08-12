import numpy as np


class Statistics:
    def __init__(self, best_params, mean, minimum, maximum, std, use_max):
        self.use_max = use_max
        self.min = minimum
        self.max = maximum
        self.mean = mean
        self.std = std
        self.best_params = best_params

    @property
    def significance(self):
        if not self.use_max:
            return (self.mean - self.min) / self.std
        else:
            return (self.max - self.mean) / self.std


class Periodogram:
    def __init__(self, periodogram, params, use_max):
        self.use_max = use_max
        self.data = periodogram
        self.params = params

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

    def best_params(self, n=1):
        if not self.use_max:
            partition = np.argpartition(self.data, n)[:n]
        else:
            partition = np.argpartition(self.data, len(self.data) - n)[:-n]
        
        idxs = np.unravel_index(partition, self.data.shape)
        values = self.data[idxs]
        significances = np.abs(self.mean - values) / self.std

        best = []
        for (idx, val, sig) in zip(idxs, values, significances):
            params = (self.params[i][idx[i]] for i in range(len(self.params)))
            best.append({
                'value': val,
                'significance': sig,
                'params': params,
            })

        return best
