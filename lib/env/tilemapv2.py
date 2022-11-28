import time
import math
import itertools
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from .tilemap import Ordinal, TilemapConstraints
from .spaces import MaskedDiscrete, MaskedMultiDiscrete
from .utils import index_where, inbounds

def dec2seq(num, base):
    result = []
    while num > 0:
        v = int(num % base) - 1
        if v < 0:
            carryover = abs(v)
            v += base
        else:
            carryover = 0
        result.append(v)
        num //= base
        num -= carryover
    return result

def seq2dec(seq, base):
    #return sum([(s+1) * (base ** (len(seq) - i - 1) ) for i, s in enumerate(seq)])
    return sum([(s+1) * (base ** i) for i, s in enumerate(seq)])

class TilemapEnv(gym.Env):
    def __init__(self, size, constraints, initial_state=0, images=None):
        self.size = size
        self.constraints = constraints
        self.images = images

        if self.images:
            tile_size = None
            for tile in images:
                tile_size = tile_size or images[tile].shape
                assert images[tile].shape == tile_size
            self.tile_size = tile_size

        base = len(constraints.tiles)

        self.action_space = spaces.Discrete(base)
        max_seq = [base - 1 for _ in range(size[0] * size[1])]
        self.observation_space = spaces.Discrete(seq2dec(max_seq, base)+1)
        self.reset(initial_state)

    def reset(self, initial_state=0):
        self.state = initial_state
        return self.state, {}

    def deserialize_state(self, state=None):
        if state is None:
            state = self.state

        seq = dec2seq(state, len(self.constraints.tiles))
        
        expected_len = self.size[0] * self.size[1]
        state = np.array(seq, dtype=np.int32)
        state = np.pad(state, (0, expected_len - len(state)), constant_values=(-1))
        state = np.reshape(state, self.size)

        return state

    def step(self, action):
        assert action in self.action_space

        base = len(self.constraints.tiles)

        seq = dec2seq(self.state, base)
        seq.append(action)
        self.state = seq2dec(seq, base)
        assert dec2seq(self.state, base) == seq

        reward = 0
        terminated = False
        truncated = False

        if len(seq) == self.size[0] * self.size[1]:
            terminated = True
            reward = self.calculate_reward()

        return self.state, reward, terminated, truncated, {}

    def calculate_reward(self, state=None):
        if state is None:
            state = self.state

        if type(state) != np.ndarray:
            state = self.deserialize_state(state)

        invalid_count = 0
        for y, x in np.ndindex(self.size):
            ti = state[y, x]
            if ti < 0: continue
            tile = self.constraints.tiles[ti]
            for ordinal in Ordinal:
                offset = ordinal.offset()
                srd = (y + offset[0], x + offset[1])
                if not inbounds(srd, self.size): continue
                srd_tile = self.constraints.tiles[state[srd]]
                if tile not in self.constraints.constraints[srd_tile][ordinal.opposite()]:
                    invalid_count += 1
                    break

        n = self.size[0] * self.size[1]
        rw = (n - invalid_count) / n
        assert rw >= 0 and rw <= 1.0
        return rw

    def render(self, highlight=False, fig=None, ax=None):
        state = self.deserialize_state()
        print(state)

        img = np.zeros((state.shape[0] * self.tile_size[0], state.shape[1] * self.tile_size[1], self.tile_size[2]))
        
        for i, (y, x) in enumerate(np.ndindex(state.shape)):
            tile_index = state[y, x]
            
            y = i // state.shape[1]
            x = i % state.shape[1]

            y1 = y * self.tile_size[0]
            y2 = y1 + self.tile_size[0]
            x1 = x * self.tile_size[1]
            x2 = x1 + self.tile_size[1]

            if tile_index < 0:
                pass
                # if highlight and len(self.possibles[y, x]) == 0:
                #     img[y1:y2, x1:x2, :] = (1.0, 0.0, 0.0)
            else:
                img[y1:y2, x1:x2, :] = self.images[self.constraints.tiles[tile_index]]

        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.gca()

        return ax.imshow(img)

    def close(self):
        pass # Nothing to do
