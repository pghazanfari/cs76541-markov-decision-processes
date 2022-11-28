import random
import time
import numpy as np

class PolicyIteration:
    def __init__(self, mdp, gamma=0.9):
        self.mdp = mdp
        self.gamma = gamma
        self.values = {}
        self.policy = {}

        for state in mdp.states:
            actions = mdp.actions(state)
            if len(actions) > 0:
                self.policy[state] = random.choice(actions) #actions[0]

    def policy_evaluation_step(self):
        delta = 0
        for state in self.mdp.states:
            policy = self.policy.get(state)
            if policy is None: continue

            old_val = self.values.get(state, 0)

            val = 0
            for transition_prob, next_state, reward, done in self.mdp.next_states(state, policy):
                val += transition_prob * (reward + self.gamma * self.values.get(next_state, 0))

            self.values[state] = val

            delta = max(delta, abs(val - old_val))
        return delta

    def policy_evaluation(self, metrics, min_delta=0.01, max_steps=None):
        assert max_steps is None or max_steps > 0

        i = 0
        while max_steps is None or i < max_steps:
            delta = self.policy_evaluation_step()
            i += 1
            if delta < min_delta:
                break
            
        metrics['policy_evaluation']['iterations'].append(i)
        metrics['policy_evaluation']['deltas'].append(delta)
        return delta

    def policy_improvement(self):
        policy_stable = True
        for state in self.mdp.states:
            current_val = self.values.get(state, 0)
            for action in self.mdp.actions(state):
                val = 0
                for transition_prob, next_state, reward, done in self.mdp.next_states(state, action):
                    val += transition_prob * (reward + self.gamma * self.values.get(next_state, 0))
                
                if val > current_val and action != self.policy.get(state, self.mdp.actions(state)[0]):
                    self.policy[state] = action
                    current_val = val
                    policy_stable = False

        return policy_stable

    def run(self, min_delta=0.01, max_steps=None, metrics=None, evals_per_it=0):
        if metrics is None:
            metrics = {
                'policy_evaluation': {
                    'iterations': [],
                    'deltas': []
                },
                'times': [],
                'avg_rewards': []
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
            if evals_per_it > 0:
                metrics['avg_rewards'].append(np.mean(self.evaluate(evals_per_it)[0]))
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
            if type(state) == np.ndarray:
                state = state.tobytes()

            done = False
            total_reward = 0
            while not done:
                default_action = self.mdp.actions(state)[0]
                action = self.policy.get(state, default_action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                if type(next_state) == np.ndarray:
                    next_state = next_state.tobytes()

                total_reward += reward
                state = next_state
                done = terminated or truncated
            et = time.time()
            times.append(et - st)
            rewards.append(total_reward)


        return rewards, times