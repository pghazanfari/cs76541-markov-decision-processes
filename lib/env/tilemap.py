import time
import itertools
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from .spaces import MaskedDiscrete, MaskedMultiDiscrete
from .utils import index_where, inbounds

class Ordinal(Enum):
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3

    def from_char(char):
        if char[0] == 'n' or char[0] == 'N':
            return Ordinal.NORTH
        elif char[0] == 'e' or char[0] == 'E':
            return Ordinal.EAST
        elif char[0] == 's' or char[0] == 'S':
            return Ordinal.SOUTH
        elif char[0] == 'w' or char[0] == 'W':
            return Ordinal.WEST
        else:
            raise Exception(f'Invalid ordinal character: {char}')

    # (y, x)
    def offset(self):
        if self == Ordinal.NORTH:
            return (-1, 0)
        elif self == Ordinal.EAST:
            return (0, 1)
        elif self == Ordinal.SOUTH:
            return (1, 0)
        elif self == Ordinal.WEST:
            return (0, -1)
        else:
            raise NotImplementedError

    def opposite(self):
        if self == Ordinal.NORTH:
            return Ordinal.SOUTH
        elif self == Ordinal.EAST:
            return Ordinal.WEST
        elif self == Ordinal.SOUTH:
            return Ordinal.NORTH
        elif self == Ordinal.WEST:
            return Ordinal.EAST
        else:
            raise NotImplementedError

class TilemapConstraints:
    def __init__(self, tiles, constraints):
        self.tiles = tiles
        self.constraints = constraints
        self.tile_indices = {}

        for i, tile in enumerate(self.tiles):
            assert isinstance(tile, str)
            self.tile_indices[tile] = i
            if tile not in self.constraints:
                self.constraints[tile] = {}
            for ordinal in Ordinal:
                assert isinstance(self.constraints[tile], dict)
                if ordinal not in self.constraints[tile]:
                    self.constraints[tile][ordinal] = set()

    # Rules are in the form of (tile, ordinal, tile)
    def from_rules(rules):
        unique_tiles = set()
        tiles = []
        constraints = {}
        for rule in rules:
            assert len(rule) == 3
            tile1, ordinal, tile2 = rule

            # Add tiles
            for tile in [tile1, tile2]:
                if tile not in unique_tiles:
                    tiles.append(tile)
                    unique_tiles.add(tile)

            constraints[tile1] = constraints.get(tile1, {})
            constraints[tile2] = constraints.get(tile2, {})

            ordinal = [Ordinal.from_char(c) for c in ordinal]

            for o in ordinal:
                constraints[tile1][o] = constraints[tile1].get(o, set())
                constraints[tile2][o.opposite()] = constraints[tile2].get(o.opposite(), set())
                constraints[tile1][o].add(tile2)
                constraints[tile2][o.opposite()].add(tile1)

        return TilemapConstraints(tiles, constraints)

    def stringify(self, state):
        return [self.tiles[s] for s in state]

class TilemapEnv(gym.Env):
    reward_modes = ['binary', 'progressive']

    def __init__(self, size, constraints, reward_mode='binary', truncate=True, initial_state=None, images=None):
        assert reward_mode in TilemapEnv.reward_modes

        self.size = size
        self.constraints = constraints
        self.images = images
        self.reward_mode = reward_mode
        self.truncate = truncate

        if self.images:
            tile_size = None
            for tile in images:
                tile_size = tile_size or images[tile].shape
                assert images[tile].shape == tile_size
            self.tile_size = tile_size

        # For WFC, tile location is automatically selected.
        self.action_space = MaskedMultiDiscrete([*self.size, len(constraints.tiles)])
        self.observation_space = gym.spaces.MultiDiscrete([*self.size, len(constraints.tiles)])
        self.reset(state=initial_state)

    def _min_entropy(self):
        entropy = None
        for y, x in np.ndindex(self.possibles.shape):
            if self.selected[y, x]: continue
            if entropy is None or len(self.possibles[y, x]) < entropy:
                if self.truncate or len(self.possibles[y, x]) > 0:
                    entropy = len(self.possibles[y, x])
        return entropy

    def step(self, action):
        assert not self.done, f"state={self.state}, selected={self.selected}, truncations={self.truncations}"
        assert len(action) == 3, f"Invalid action {action}"
        assert action in self.action_space, f"Action {action} not in action space: {self.action_space.mask}"

        y, x, tile_index = action
        self.state[y, x] = tile_index
        self.selected[y, x] = True
        self.action_space.mask[y, x] = False

        reward = 0.0

        terminated = False
        truncated = False
                
        # Adjust possibles for surrounding tiles
        tile = self.constraints.tiles[tile_index]
        for ordinal in Ordinal:
            offset = ordinal.offset()
            srd = (y + offset[0], x + offset[1])
            if not inbounds(srd, self.state.shape): continue
            constraints = self.constraints.constraints[tile][ordinal]
            if len(constraints) > 0: # Skip empty sets
                self.possibles[srd] = self.possibles[srd].intersection(constraints)

            if not self.selected[srd] and len(self.possibles[srd]) == 0:
                truncated = self.truncate
                self.truncations[srd] = True

        min_entropy = self._min_entropy()
        
        if min_entropy == 0:
            assert truncated or terminated

        for y, x in np.ndindex(self.state.shape):
            if self.selected[y, x] or len(self.possibles[y, x]) != min_entropy:
                self.action_space.mask[y, x, :] = False
            else:
                self.action_space.mask[y, x] = [t in self.possibles[y, x] for t in self.constraints.tiles]

        if self.truncate:
            assert reward == 0.0
            if np.all(self.selected == True):
                terminated = True
                reward = 1.0
        else:
            if np.all(np.logical_or(self.selected == True, self.truncations == True)):
                terminated = True
                if self.reward_mode == 'binary':
                    reward = 0.0
                else:
                    reward = np.count_nonzero(self.selected) / (self.size[0] * self.size[1])


        self.done = terminated or (self.truncate and truncated)

        info = {}

        #state reward terminated, truncated, info
        return self.state, reward, terminated, (self.truncate and truncated), info

    def reset(self, state=None):
        self.done = False
        self.state = np.full(self.size, -1, dtype=np.int32) if state is None else np.copy(state) # Copy to avoid frombuffer immutability
        assert self.state.dtype == np.int32

        self.selected = np.full(self.size, False, dtype=bool)
        self.selected[self.state > -1] = True
        self.done = self.selected.all()

        self.truncations = np.full(self.size, False, dtype=bool)

        self.possibles = np.full(self.size, None, dtype=np.object)
        for y, x in np.ndindex(self.size):
            if self.selected[y, x]:
                self.possibles[y, x] = set()
            else:
                possible = set(self.constraints.tiles)
                for ordinal in Ordinal:
                    offset = ordinal.offset()
                    srd = (y + offset[0], x + offset[1])
                    if srd[0] < 0 or srd[1] < 0 or srd[0] >= self.state.shape[0] or srd[1] >= self.state.shape[1]: continue
                    if self.state[srd] >= 0:
                        tile = self.constraints.tiles[self.state[srd]]
                        constraints = self.constraints.constraints[tile][ordinal.opposite()]
                        possible = possible.intersection(constraints)

                self.possibles[y, x] = possible
                if len(possible) == 0:
                    self.done = self.truncate
                    self.truncations[y, x] = True
            
        # self.action_space.mask[...] = False
        # for y, x in np.ndindex(self.size):
        #     self.action_space.mask[y, x] = [t in self.possibles[y, x] for t in self.constraints.tiles]

        min_entropy = self._min_entropy()
        for y, x in np.ndindex(self.state.shape):
            if self.selected[y, x] or len(self.possibles[y, x]) != min_entropy:
                self.action_space.mask[y, x, :] = False
            else:
                self.action_space.mask[y, x] = [t in self.possibles[y, x] for t in self.constraints.tiles]


        return self.state, {}

    def render(self, highlight=False, fig=None, ax=None):
        state = self.state
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
                if highlight and len(self.possibles[y, x]) == 0:
                    img[y1:y2, x1:x2, :] = (1.0, 0.0, 0.0)
            else:
                img[y1:y2, x1:x2, :] = self.images[self.constraints.tiles[tile_index]]

        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.gca()

        return ax.imshow(img)

    def close(self):
        pass # Nothing to do