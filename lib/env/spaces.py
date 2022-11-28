import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .utils import index_where

class MaskedDiscrete(spaces.Discrete):
    def __init__(self,  n, seed=None):
        super().__init__(n, seed=seed)
        self.mask = set(range(0, n))

    def sample(self):
        return self.np_random.choice(np.array(self.mask))

    def contains(self, x):
        return super().contains(x) and int(x) in self.mask

    def __repr__(self):
        return "MaskedDiscrete(%d)" % self.n

    def __eq__(self, other):
        return super().__eq__(other) and self.mask == other.mask

class MaskedMultiDiscrete(spaces.MultiDiscrete):
    def __init__(self, nvec, dtype=np.int64, seed=None):
        super().__init__(nvec, dtype=dtype, seed=seed)
        self.mask = np.full(tuple(nvec), False, dtype=bool)

    def sample(self):
        l = list(index_where(self.mask == True))
        idx = self.np_random.choice(range(len(l)))
        return l[idx]

    def contains(self, x):
        return super().contains(list(x)) and self.mask[tuple(x)]

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

    def __repr__(self):
        return "MaskedMultiDiscrete({})".format(self.nvec)

    def __getitem__(self, index):
        nvec = self.nvec[index]
        if nvec.ndim == 0:
            subspace = MaskedDiscrete(nvec)
            subspace.mask = set(index_where(self.mask == True))
            raise NotImplementedError # TODO: need to double check this
            #subspace = Discrete(nvec)
        else:
            subspace = MaskedMultiDiscrete(nvec, self.dtype)
            subspace.mask = self.mask.copy()
        subspace.np_random.set_state(self.np_random.get_state())  # for reproducibility
        return subspace

    def __len__(self):
        if self.nvec.ndim >= 2:
            logger.warn("Get length of a multi-dimensional MultiDiscrete space.")
        return len(self.nvec)

    def __eq__(self, other):
        return super().__eq__(other) and self.mask == other.mask