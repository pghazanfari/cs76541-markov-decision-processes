from ..mdp.tilemapv3 import TilemapMDP
from ..mdp.frozen_lake import FrozenLakeMDP
from ..mdp.value_iteration import ValueIteration
from ..mdp.policy_iteration import PolicyIteration
from ..mdp.q_learning  import QLearning
from .tilemap import TILEMAP_CONSTRAINTS, TILEMAP_IMAGES

from .utils import gridsearch

import matplotlib.pyplot as plt
import numpy as np
import time

class Experiment:
    mdps = {
        'tilemap': TilemapMDP((2, 2), TILEMAP_CONSTRAINTS, TILEMAP_IMAGES),
        'frozen_lake': FrozenLakeMDP()
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = [{} for _ in Experiment.step_fns]
        self.metrics = [{} for _ in Experiment.step_fns]

    def run_step(self, step, render=True, update_state=True, **kwargs):
        assert step >= 0 and step < len(Experiment.step_fns)
        if update_state:
            self.state[step] = {**self.state[step], **kwargs}
        state = self.state[step]
        metrics = self.metrics[step]
        
        step_fn = Experiment.step_fns[step]
        render_fn = None

        if type(step_fn) == tuple:
            step_fn, render_fn = step_fn

        step_fn(self, state, metrics)
        if render and render_fn is not None:
            render_fn(self, state, metrics)

    def run(self, render=True, update_state=True, **kwargs):
        for i in range(len(Experiment.step_fns)):
            self.run_step(i, render=render, update_state=update_state, **kwargs)

    def discover_best_gamma_for_value_iteration(self, state, metrics):
        for mdp in Experiment.mdps:
            if state.get(mdp, {}).get('gamma') is None:
                state[mdp] = state.get(mdp, {})

                def eval_fn(gamma):
                    vi = ValueIteration(Experiment.mdps[mdp], gamma)
                    return vi.run(evals_per_it=10)['avg_rewards'][-1]

                best_iters, best_params = gridsearch({'gamma': np.linspace(0.0, 1.0, num=10)}, eval_fn)
                state[mdp]['gamma'] = best_params['gamma'] 
        print("value_iteration: ", state)

    def discover_best_gamma_for_policy_iteration(self, state, metrics):
        for mdp in Experiment.mdps:
            if state.get(mdp, {}).get('gamma') is None:
                state[mdp] = state.get(mdp, {})

                def eval_fn(gamma):
                    pi = PolicyIteration(Experiment.mdps[mdp], gamma)
                    return pi.run(evals_per_it=10)['avg_rewards'][-1]

                best_iters, best_params = gridsearch({'gamma': np.linspace(0.0, 1.0, num=10)}, eval_fn)
                state[mdp]['gamma'] = best_params['gamma'] 
        print("policy_iteration: ", state)

    def value_iteration(self, state, metrics):
        for mdp in Experiment.mdps:
            evals_per_it = state.get(mdp, {}).get('evals_per_it', 10)

            default_gamma = self.state[0].get(mdp, {}).get('gamma')
            gamma = state.get(mdp, {}).get('gamma', default_gamma)
            vi = ValueIteration(Experiment.mdps[mdp], gamma=gamma)
            metrics[mdp] = vi.run(evals_per_it=evals_per_it)
            
    def policy_iteration(self, state, metrics):
        for mdp in Experiment.mdps:
            evals_per_it = state.get(mdp, {}).get('evals_per_it', 10)
            default_gamma = self.state[2].get(mdp, {}).get('gamma')
            gamma = state.get(mdp, {}).get('gamma', default_gamma)
            pi = PolicyIteration(Experiment.mdps[mdp], gamma=gamma)
            metrics[mdp] = pi.run(evals_per_it=evals_per_it)

    def value_and_policy_iteration(self, state, metrics):
        metrics['value_iteration'] = self.metrics[1]
        metrics['policy_iteration'] = self.metrics[3]

    def render_value_and_policy_iteration(self, state, metrics):
        for mdp in Experiment.mdps:
            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            fig.suptitle(mdp)

            for i, algo in enumerate(metrics):
                ax = axes[i]
                ax.set_title(algo)
                ax.plot(metrics[algo][mdp]['avg_rewards'], label='Avg Reward')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Avg Reward')
                rightax = ax.twinx()
                rightax.plot(np.cumsum(metrics[algo][mdp]['times']), label='Time', color='r')
                rightax.legend()
                rightax.set_ylabel("Time (s)")
        fig.show()

    def discover_best_params_for_q_learning(self, state, metrics):
         for mdp in Experiment.mdps:
            gamma = state.get(mdp, {}).get('gamma')
            lr = state.get(mdp, {}).get('lr')
            eps_decay = state.get(mdp, {}).get('eps_decay')

            default_max_epochs = self.state[6].get(mdp, {}).get('max_epochs', 1000)
            max_epochs = state.get(mdp, {}).get('max_epochs', default_max_epochs)

            default_nlast = self.state[6].get(mdp, {}).get('nlast', 10)
            nlast = state.get(mdp, {}).get('nlast', default_nlast)

            assert nlast < max_epochs

            if gamma is None or lr is None or eps_decay is None:
                params = {
                    'gamma': [0, 0.25, 0.5, 0.75, 1.0] if gamma is None else [gamma],
                    'lr': [0.1, 0.25, 0.5, 0.75, 1.0] if lr is None else [lr],
                    'eps_decay': [0.9, 0.99, 0.999] if eps_decay is None else [eps_decay]
                }

                def eval_fn(gamma, lr, eps_decay):
                    qlearning = QLearning(Experiment.mdps[mdp], lr=lr, gamma=gamma, eps_decay=eps_decay)
                    rewards = [qlearning.step() for _ in range(max_epochs)]
                    avg_reward = np.mean(rewards[-nlast:])
                    return avg_reward

                best_avg_reward, best_params = gridsearch(params, eval_fn)
                print(f"{mdp}: avg_reward={best_avg_reward}, params={best_params}")
                state[mdp] = state.get(mdp, {})
                state[mdp]['gamma'] = best_params['gamma']
                state[mdp]['lr'] = best_params['lr']
                state[mdp]['eps_decay'] = best_params['eps_decay']

    def q_learning(self, state, metrics):
        for mdp in Experiment.mdps:
            max_epochs = state.get(mdp, {}).get('max_epochs', 1000)
            nlast = state.get(mdp, {}).get('nlast', 100)
            min_delta = state.get(mdp, {}).get('min_delta', 0.01)

            default_gamma = self.state[5].get(mdp, {}).get('gamma')
            gamma = state.get(mdp, {}).get('gamma', default_gamma)

            default_lr = self.state[5].get(mdp, {}).get('lr')
            lr = state.get(mdp, {}).get('lr', default_lr)

            default_eps_decay = self.state[5].get(mdp, {}).get('eps_decay')
            eps_decay = state.get(mdp, {}).get('eps_decay', default_eps_decay)

            print(f"{mdp}: lr={lr}, gamma={gamma}, eps_decay={eps_decay}")
            qlearning = QLearning(Experiment.mdps[mdp], lr=lr, gamma=gamma, eps_decay=eps_decay)
            
            metrics[mdp] = {
                'step_times': []
            }

            rewards = []
            avg_rewards = []
            st = time.time()
            for i in range(max_epochs):
                step_st = time.time()
                rewards.append(qlearning.step())
                step_et = time.time()
                metrics[mdp]['step_times'].append(step_et - step_st)
                nlast_rewards = rewards[-min(nlast, len(rewards)):]
                avg_rewards.append(np.mean(nlast_rewards))
                
                # nlast_avg_rewards = avg_rewards[-min(nlast, len(avg_rewards)):]
                # if i > nlast and np.max(nlast_avg_rewards) - np.min(nlast_rewards) < min_delta:
                #     print(f"{mdp}: delta threshold reached")
                #     break

            et = time.time()
            metrics[mdp]['rewards'] = rewards
            metrics[mdp]['avg_rewards'] = avg_rewards
            metrics[mdp]['time'] = et - st

    def render_q_learning(self, state, metrics):
        for mdp in Experiment.mdps:
            plt.figure()
            plt.plot(metrics[mdp]['avg_rewards'], label='Avg Rewards')

            plt.gca().set_ylabel('Reward')
            plt.gca().set_xlabel('Epoch')

            rightax = plt.gca().twinx()
            rightax.plot(np.cumsum(metrics[mdp]['step_times']), label='Cumulative Time (s)', color='r')
            rightax.set_ylabel('Time (s)')
            rightax.legend()

            plt.title(f"Q Learning ({mdp})")
            plt.gca().legend()
            plt.show()

            final_reward = metrics[mdp]['avg_rewards'][-1]
            print(f"{mdp}: time={metrics[mdp]['time']}, final_reward={final_reward}")

    step_fns = [
        discover_best_gamma_for_value_iteration,
        value_iteration,
        discover_best_gamma_for_policy_iteration,
        policy_iteration,
        (value_and_policy_iteration, render_value_and_policy_iteration),
        discover_best_params_for_q_learning,
        (q_learning, render_q_learning)
    ]