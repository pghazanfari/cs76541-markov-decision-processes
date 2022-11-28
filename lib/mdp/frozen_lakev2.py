import gymnasium as gym
import numpy as np

class FrozenLakeMDP:
    def __init__(self):
        self._env = self.env
        self.transition_probs = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n))
        self._rewards = np.zeros(self.transition_probs.shape)
        self._next_states = np.zeros(self.transition_probs.shape, dtype=np.int32)

        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                for transition_prob, next_state, reward, done in self._env.P[state][action]:
                    self.transition_probs[state, action, next_state] = transition_prob
                    self._rewards[state, action, next_state] = reward
                    self._next_states[state, action, next_state] = next_state

    @property
    def env(self):
        return gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='rgb_array')

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def next_states(self):
        return self._next_states

    @property
    def transition_probabilities(self):
        return self.transition_probs

    @property
    def rewards(self):
        return self._rewards