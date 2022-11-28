import random
import time
import numpy as np

class PolicyIteration:
    def __init__(self, mdp, gamma=0.9):
        self.mdp = mdp
        self.gamma = gamma
        self.values = np.zeros((mdp.observation_space.n,))
        self.policy = np.random.randint(0, mdp.action_space.n, size=(mdp.observation_space.n,))
        #self.policy = np.zeros((mdp.observation_space.n,), dtype=np.int32)
        print(f"policy: {self.policy}")

    def policy_evaluation_step(self):
        v = self.values

        v_ns = np.take(self.values, self.mdp.next_states)
        v_ns = np.take(v_ns, self.policy, axis=1)

        tp = np.take(self.mdp.transition_probs, self.policy, axis=1)
        rw = np.take(self.mdp.rewards, self.policy, axis=1)

        v_s = tp * (rw + self.gamma * v_ns)
        v_s = v_s.sum(axis=-1)

        self.values = v_s

        delta = np.max(np.abs(self.values - v))
        return delta

    def policy_evaluation(self, metrics, min_delta=1e-5, max_steps=None):
        assert max_steps is None or max_steps > 0

        i = 0
        while max_steps is None or i < max_steps:
            delta = self.policy_evaluation_step()
            print(f"delta={delta}")
            i += 1
            if delta < min_delta:
                break
            
        metrics['policy_evaluation']['iterations'].append(i)
        metrics['policy_evaluation']['deltas'].append(delta)
        return delta

    def policy_improvement(self):
        policy = self.policy

        v_ns = np.take(self.values, self.mdp.next_states)
        v_s = self.mdp.transition_probs * (self.mdp.rewards + self.gamma * v_ns)
        v_s = v_s.sum(axis=-1)
        self.policy = np.argmax(v_s, axis=-1)
        print(f"policy: {self.policy}")

        return np.array_equal(policy, self.policy)

    def run(self, min_delta=0.01, max_steps=None, metrics=None):
        if metrics is None:
            metrics = {
                'policy_evaluation': {
                    'iterations': [],
                    'deltas': []
                },
                'times': []
            }

        policy_stable = False
        epochs = 0
        st = time.time()
        while not policy_stable:
            iter_st = time.time()
            self.policy_evaluation(metrics, min_delta, max_steps)
            policy_stable = self.policy_improvement()
            iter_et = time.time()
            metrics['times'].append(iter_et - iter_st)
            epochs += 1
        et = time.time()
        metrics['epochs'] = epochs
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