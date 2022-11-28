import gymnasium as gym

class FrozenLakeMDP:
    def __init__(self):
        self._env = self.env

    @property
    def env(self):
        return gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    @property
    def states(self):
        return range(self._env.observation_space.n)

    def actions(self, state):
        return range(4)

    def next_states(self, state, action):
        return self._env.P[state][action]