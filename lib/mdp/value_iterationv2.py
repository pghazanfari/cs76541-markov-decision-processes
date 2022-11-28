import time
import numpy as np
import matplotlib.pyplot as plt

class ValueIteration:
    def __init__(self, mdp, gamma=0.9):
        self.mdp = mdp
        self.gamma = gamma
        self.values = np.zeros((mdp.observation_space.n,))
        self.policy = np.zeros((mdp.observation_space.n,))

    def step(self):
        v = self.values
        # (state, action, state)

        v_ns = np.take(self.values, self.mdp.next_states)
        v_s = self.mdp.transition_probs * (self.mdp.rewards + self.gamma * v_ns)
        v_s = v_s.sum(axis=-1)

        self.values = v_s.max(axis=-1)
        self.policy = v_s.argmax(axis=-1)

        delta = np.max(np.abs(self.values - v))

        return delta

    def run(self, min_delta=1e-5, max_steps=None, metrics=None):
        assert max_steps is None or max_steps > 0

        if metrics is None:
            metrics = {
                'times': [],
                'deltas': []
            }

        i = 0
        st = time.time()
        while max_steps is None or i < max_steps:
            step_st = time.time()
            delta = self.step()
            step_et = time.time()
            #print(f"step_time={step_et - step_st}, delta={delta}")
            metrics['times'].append(step_et - step_st)
            metrics['deltas'].append(delta)
            i += 1
            if delta < min_delta:
                break
        et = time.time()
        metrics['iterations'] = i
        metrics['time'] = et - st

        return metrics

    def evaluate(self, epochs=1):
        rewards = []
        times = []

        for i in range(epochs):
            st = time.time()
            env = self.mdp.env
            state, _ = env.reset()

            done = False
            total_reward = 0
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(self.policy[state])
                total_reward += reward
                state = next_state
                done = terminated or truncated
            et = time.time()
            times.append(et - st)
            rewards.append(total_reward)


        return rewards, times


