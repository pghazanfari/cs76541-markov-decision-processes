import itertools
import time
from ..env.tilemap import Ordinal
from ..env.tilemapv2 import TilemapEnv, dec2seq, seq2dec
from ..env.utils import index_where

import numpy as np

class TilemapMDP:
    def __init__(self, size, constraints, images=None):
        self.size = size
        self.constraints = constraints
        self.images = images
        
        base = len(self.constraints.tiles) 
        seq = [base - 1 for _ in range(self.size[0] * self.size[1])]
        self._states = range(seq2dec(seq, base))
        cutoff_seq = [base - 1 for _ in range(self.size[0] * self.size[1] - 1)]
        self.terminal_cutoff = seq2dec(cutoff_seq, base)
        self._states = range(self.terminal_cutoff+1)

        self._cache = {}
        st = time.time()
        for i, state in enumerate(self.states):
            for action in self.actions(state):
                env = TilemapEnv(self.size, self.constraints, initial_state=state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self._cache[(state, action)] = (1.0, next_state, reward, terminated or truncated)

    @property
    def env(self):
        return TilemapEnv(self.size, self.constraints, images=self.images)

    @property
    def states(self):
        return self._states

    def actions(self, state):
        return range(len(self.constraints.tiles))

    def next_states(self, state, action=None):
        if state > self.terminal_cutoff:
            return

        if action is not None:
            actions = [action]
        else:
            actions = self.actions(state)

        for action in actions:
            yield self._cache[(state, action)]