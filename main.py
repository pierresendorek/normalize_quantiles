import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(42)

def sample():
    if np.random.rand() < 0.5:
        return np.random.normal() - 3
    else:
        return np.random.normal() + 3
    
nb_samples = 10**4
nb_quantiles = 10**3

samples = [sample() for _ in range(nb_samples)]
samples_2 = [sample() for _ in range(nb_samples)]


class UniformNormalizer:
    def fit(self, samples, nb_quantiles):
        q = np.linspace(0, 1, nb_quantiles)
        self.quantiles = np.quantile(samples, q)
        self.f = interp1d(self.quantiles, q, kind='linear', fill_value='extrapolate')
        return self

    def transform(self, samples):
        normalized = self.f(samples)
        # Ensure values are within [0, 1]
        return np.clip(normalized, 0, 1)


inverse_quantiles = UniformNormalizer().fit(samples, nb_quantiles).transform(samples_2)


plt.plot(np.quantile(inverse_quantiles, np.linspace(0, 1, 1000))) # It is the identity function on [0,1]
plt.show()

