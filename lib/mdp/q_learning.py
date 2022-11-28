import random
import numpy as np
from collections import defaultdict

from ..env.utils import index_where

class QLearning:
    def __init__(self, mdp, lr, gamma, eps=1.0, eps_decay=0.999):
        self.mdp = mdp
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))

    def policy(self, state):        
        best_actions = []
        best_q = 0
        for action in self.mdp.actions(state):
            if self.Q[state][action] > best_q:
                best_actions = [action]
                best_q = self.Q[state][action]
            elif self.Q[state][action] == best_q:
                best_actions.append(action)

        if len(best_actions) == 0: return best_q, None

        #return best_q, random.choice(best_actions)
        return best_q, best_actions[0]

    def _step(self, state, env):
        if random.random() < self.eps:
            action = env.action_space.sample()
        else:
            _, action = self.policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Hack to convert states to a hashable type
        if type(next_state) is np.ndarray:
            next_state = next_state.tobytes()

        q_next, _ = self.policy(next_state)
        try:
            self.Q[state][action] += self.lr * (reward + self.gamma * q_next - self.Q[state][action])
        except:
            print(f"self.Q[state]={self.Q[state]}")
            print(f"self.Q[state][action]={self.Q[state][action]}")
            raise

        #self.Q[state][action] = (1.0 - self.lr) * self.Q[state].get(action, 0) + self.lr * (reward + self.gamma * q_next)

        return next_state, reward, terminated or truncated

    def step(self):
        env = self.mdp.env
        state, _ = env.reset()
        if type(state) == np.ndarray:
            state = state.tobytes()

        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = self._step(state, env)
            total_reward += reward
            state = next_state

        self.eps = self.eps * self.eps_decay
        return reward

    def evaluate(self, epochs=1):
        rewards = []
        times = []

        for i in range(epochs):
            st = time.time()
            env = self.mdp.env
            state, _ = env.reset()
            if type(state) == np.ndarray:
                state = state.tobytes()

            done = False
            total_reward = 0
            while not done:
                action = self.policy.get(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                if type(next_state) == np.ndarray:
                    next_state = next_state.tobytes()

                total_reward += reward
                state = next_state
            et = time.time()
            times.append(et - st)
            rewards.append(total_reward)


        return rewards, times