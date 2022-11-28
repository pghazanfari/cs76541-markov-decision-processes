import numpy as np

def index_where(x):
    for a in np.flatnonzero(x):
        yield tuple([int(b) for b in np.unravel_index(a, x.shape)])

def inbounds(idx, shape):
    assert len(idx) == len(shape)
    for i in range(len(idx)):
        if idx[i] < 0 or idx[i] >= shape[i]:
            return False
    return True