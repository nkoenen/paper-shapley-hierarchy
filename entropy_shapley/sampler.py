"""Background samplers for the marginal Shapley imputation path.

Both samplers return a (sample_size, p) array of background feature vectors
that the HierarchyImputer averages over.
"""

import numpy as np


class MarginalTabularSampler:
    """Draws rows i.i.d. from background data (marginal distribution).

    Each call returns a fresh random subsample, so repeated calls to
    HierarchyImputer give independent MC estimates.
    """

    def __init__(self, data: np.ndarray, sample_size: int | None = None):
        self.data        = data
        self.sample_size = sample_size or len(data)

    def __call__(self) -> np.ndarray:
        idx = np.random.choice(len(self.data), size=self.sample_size, replace=False)
        return self.data[idx]


class BaselineTabularSampler:
    """Always returns a constant baseline value for all features.

    Useful as a deterministic baseline (e.g. all zeros or feature means).
    """

    def __init__(self, p: int, value: float | np.ndarray = 0.0):
        self.p = p

        if np.isscalar(value):
            self.value = np.full(p, value)
        else:
            assert len(value) == p, "Value vector must match number of features"
            self.value = np.asarray(value)

    def __call__(self) -> np.ndarray:
        return np.tile(self.value, (1, 1))
