import itertools
from ..env.tilemapv2 import TilemapEnv, dec2seq, seq2dec
from ..env.tilemap import Ordinal
from ..env.utils import index_where

import numpy as np

class TilemapMDP:
    def __init__(self, size, constraints, images=None):
        self.size = size
        self.constraints = constraints
        self.images = images

        env = self.env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.transition_probs = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
        self._rewards = np.zeros(self.transition_probs.shape)
        self._next_states = np.zeros(self.transition_probs.shape, dtype=np.int32)

        base = len(constraints.tiles)

        print("env.observation_space.n =", env.observation_space.n)

        for state in range(env.observation_space.n):
            seq = dec2seq(state, base)
            assert len(seq) <= size[0] * size[1], f"base={base}, state={state}, seq={seq}"
            if len(seq) == size[0] * size[1]: continue
            # Skip states except for the last move
            for action in range(env.action_space.n):
                next_seq = seq + [action]
                next_state = seq2dec(next_seq, base)
                try:
                    self.transition_probs[state, action, next_state] = 1.0
                except:
                    print(f"base={base}")
                    print(f"self.transition_probs.shape={self.transition_probs.shape}")
                    print(f"state={state}, action={action}, next_state={next_state}")
                    print(f"seq={seq}, next_seq={next_seq}")
                    raise
                if len(seq) == (size[0] * size[1]) - 1:
                    self._rewards[state, action, next_state] = env.calculate_reward(next_state)
                self._next_states[state, action, next_state] = next_state
    @property
    def env(self):
        return TilemapEnv(self.size, self.constraints, images=self.images)

    @property
    def next_states(self):
        return self._next_states

    @property
    def transition_probabilities(self):
        return self.transition_probs

    @property
    def rewards(self):
        return self._rewards