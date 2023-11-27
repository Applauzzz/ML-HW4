from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from hiive.mdptoolbox.example import forest
import numpy as np

from numpy.random import choice
import pandas as pd
import seaborn as sns
np.random.seed(44)

P, R = forest(S=400, r1=10, r2=6, p=0.1)
alpha_decs = [0.99, 0.999]
alpha_mins =[0.001, 0.0001]
eps = [10.0, 1.0]
eps_dec = [0.99, 0.999]
iters = [1000000, 10000000]
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def test_policy(P, R, policy, test_count=1000, gamma=0.9):
    num_states = P.shape[-1]
    total_reward = 0

    for state in range(num_states):
        state_reward = sum(run_episode(P, R, policy, state, gamma) for _ in range(test_count))
        total_reward += state_reward

    return total_reward / (num_states * test_count)

def run_episode(P, R, policy, initial_state, gamma):
    episode_reward, disc_rate = 0, 1
    current_state = initial_state

    while True:
        action = policy[current_state]
        next_state = choice(len(P[action][current_state]), p=P[action][current_state])
        episode_reward += R[current_state][action] * disc_rate
        disc_rate *= gamma

        if next_state == 0:  # assuming episode ends when reaching state 0
            break
        current_state = next_state

    return episode_reward
import pandas as pd
from mdptoolbox.mdp import ValueIteration
def trainVI(P, R, discount=0.9, epsilon=[1e-9]):
    vi_df = pd.DataFrame(columns=["Epsilon", "Policy", "Iteration", "Time", "Reward", "Value Function"])

    for eps in epsilon:
        vi = ValueIteration(P, R, gamma=discount, epsilon=eps, max_iter=int(1e15))
        vi.run()
        reward = test_policy(P, R, vi.policy)

        vi_df = vi_df.append({
            "Epsilon": float(eps), 
            "Policy": vi.policy, 
            "Iteration": vi.iter, 
            "Time": vi.time, 
            "Reward": reward, 
            "Value Function": vi.V
        }, ignore_index=True)
    return vi_df

import pandas as pd
from mdptoolbox.mdp import QLearning

def trainQ(P, R, discount=0.9, alpha_dec=[0.99], alpha_min=[0.001], 
           epsilon=[1.0], epsilon_decay=[0.99], n_iter=[1000000]):
    q_df = pd.DataFrame(columns=["Iterations", "Alpha Decay", "Alpha Min",
                                 "Epsilon", "Epsilon Decay", "Reward",
                                 "Time", "Policy", "Value Function",
                                 "Training Rewards"])

    count = 0
    for iter_count in n_iter:
        for eps in epsilon:
            for eps_dec in epsilon_decay:
                for a_dec in alpha_dec:
                    for a_min in alpha_min:
                        q_learning_model = run_q_learning(P, R, discount, iter_count, 
                                                          a_dec, a_min, eps, eps_dec)
                        reward = test_policy(P, R, q_learning_model.policy)
                        count += 1
                        print(f"{count}: {reward}")

                        q_df = append_q_learning_results(q_df, q_learning_model, iter_count,
                                                         a_dec, a_min, eps, eps_dec, reward)

    return q_df

def run_q_learning(P, R, discount, n_iter, alpha_decay, alpha_min, epsilon, epsilon_decay):
    """Runs the Q-Learning algorithm with specified parameters."""
    q_learning = QLearning(P, R, discount, alpha_decay=alpha_decay, 
                           alpha_min=alpha_min, epsilon=epsilon, 
                           epsilon_decay=epsilon_decay, n_iter=n_iter)
    q_learning.run()
    return q_learning

def append_q_learning_results(df, model, n_iter, alpha_decay, alpha_min, epsilon, epsilon_decay, reward):
    """Appends the results of a Q-Learning model run to the dataframe."""
    training_rewards = [s['Reward'] for s in model.run_stats]
    new_row = {
        "Iterations": n_iter, "Alpha Decay": alpha_decay, "Alpha Min": alpha_min,
        "Epsilon": epsilon, "Epsilon Decay": epsilon_decay, "Reward": reward,
        "Time": model.time, "Policy": model.policy, "Value Function": model.V,
        "Training Rewards": training_rewards
    }
    return df.append(new_row, ignore_index=True)

if __name__ == "__main__":
    vi_df = trainVI(P, R, epsilon=[1e-1,1e-2, 1e-3,1e-5, 1e-6, 1e-9, 1e-12, 1e-15])
    pi = PolicyIteration(P, R, gamma=0.9, max_iter=1e6)
    pi.run()

    q_df = trainQ(P, R, discount=0.9, alpha_dec=alpha_decs, alpha_min=alpha_mins, 
                epsilon=eps, epsilon_decay=eps_dec, n_iter=iters)
