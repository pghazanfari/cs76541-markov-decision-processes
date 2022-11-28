import time
import numpy as np

class ValueIteration:
    def __init__(self, mdp, gamma=0.9):
        self.mdp = mdp
        self.gamma = gamma
        self.values = {}
        self.policy = {}

    def step(self):
        delta = 0
        for state in self.mdp.states:
            old_val = self.values.get(state, 0)
            best_val = 0
            best_action = None
            for action in self.mdp.actions(state):
                val = 0
                for transition_prob, next_state, reward, done in self.mdp.next_states(state, action):
                    val += transition_prob * (reward + self.gamma * self.values.get(next_state, 0))
                
                if best_action is None or val > best_val:
                    best_action = action
                    best_val = val

            if best_val >  self.values.get(state, 0):
                self.policy[state] = best_action

            self.values[state] = best_val

            delta = max(delta, abs(best_val - old_val))
        return delta

    def run(self, min_delta=0.01, max_steps=None, metrics=None, evals_per_it=0):
        assert max_steps is None or max_steps > 0

        if metrics is None:
            metrics = {
                'times': [],
                'deltas': [],
                'avg_rewards': []
            }

        i = 0
        st = time.time()
        while max_steps is None or i < max_steps:
            step_st = time.time()
            delta = self.step()
            step_et = time.time()
            #print(f"step_time={step_et-step_st}, delta={delta}")
            metrics['times'].append(step_et - step_st)
            metrics['deltas'].append(delta)
            if evals_per_it > 0:
                metrics['avg_rewards'].append(np.mean(self.evaluate(evals_per_it)[0]))
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


