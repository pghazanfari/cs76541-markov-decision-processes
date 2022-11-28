import itertools
from ..env.tilemap import TilemapEnv, Ordinal
from ..env.utils import index_where

import numpy as np

class TilemapMDP:
    def __init__(self, size, constraints, images=None):
        self.size = size
        self.constraints = constraints
        self.images = images
        self._actions = {}
        self._states = None
        self._cache = {}

    @property
    def env(self):
        return TilemapEnv(self.size, self.constraints, truncate=False, reward_mode='progressive', images=self.images)

    @property
    def states(self):
        if self._states is None:
            self._states = []
            possible_tiles = list(range(len(self.constraints.tiles))) + [-1]
            for flattened_tilemap in itertools.combinations_with_replacement(possible_tiles, self.size[0] * self.size[1]):
                tilemap = np.array(flattened_tilemap, dtype=np.int32)
                self._states.append(tilemap.tobytes())
        return self._states

    def actions(self, state):
        if state not in self._actions:
            st = np.frombuffer(state, dtype=np.int32).reshape(self.size)
            env = TilemapEnv(self.size, self.constraints, truncate=False, reward_mode='progressive', initial_state=st)
            self._actions[state] = list(index_where(env.action_space.mask == True))
        return self._actions[state]

    def next_states(self, state, action=None):
        if action is not None:
            actions = [action]
        else:
            actions = self.actions(state)

        for action in actions:
            key = (state, action)
            if key not in self._cache:
                st = np.frombuffer(state, dtype=np.int32).reshape(self.size)
                env = TilemapEnv(self.size, self.constraints, truncate=False, reward_mode='progressive', initial_state=st)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self._cache[key] = (1.0, next_state.tobytes(), reward, terminated or truncated)
            yield self._cache[key]