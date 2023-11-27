import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
def init_map():
    np.random.seed(0)
    four = generate_random_map(4)

    np.random.seed(1)
    eight = generate_random_map(8)

    np.random.seed(2)
    sixteen = generate_random_map(16)

    np.random.seed(44)
    tvelve = generate_random_map(12)
    MAPS = {
        "4x4": four,
        "8x8": eight,
        "12x12": tvelve,
        "16x16": sixteen
    }
    return MAPS
def test_policy(env, policy, n_epoch=1000):
    rewards = []
    episode_counts = []

    for i in range(n_epoch):
        total_reward, episode_length = run_episode(env, policy)
        rewards.append(total_reward)
        episode_counts.append(episode_length)

    mean_reward = np.mean(rewards)
    mean_episode_length = np.mean(episode_counts)
    
    return mean_reward, mean_episode_length, rewards, episode_counts
def run_episode(env, policy):
    """Runs a single episode with the given policy."""
    state = env.reset()
    total_reward = 0
    episode_length = 0

    while True:
        action = int(policy[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        episode_length += 1

        if done or episode_length >= 1000:
            break

    return total_reward, episode_length


def value_iteration(env, discount=0.9, epsilon=1e-12):
    start = timer()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    value_table = np.zeros(n_states)
    policy = np.zeros(n_states)

    max_change = epsilon + 1
    while max_change > epsilon:
        max_change = update_values_and_policy(env, n_states, n_actions, value_table, policy, discount)

    time_spent = timedelta(seconds=timer() - start)
    print(f"Solved in {np.count_nonzero(policy)} episodes and {time_spent} seconds")
    return policy, np.count_nonzero(policy), time_spent

def update_values_and_policy(env, n_states, n_actions, value_table, policy, discount):
    """Updates the value table and policy."""
    max_change = 0
    for state in range(n_states):
        old_value = value_table[state]
        value_table[state], policy[state] = calculate_best_action(env, state, n_actions, value_table, discount)
        max_change = max(max_change, abs(old_value - value_table[state]))
    
    return max_change

def calculate_best_action(env, state, n_actions, value_table, discount):
    """Finds the best action for a given state."""
    best_action = 0
    best_value = -np.inf

    for action in range(n_actions):
        expected_value = sum(prob * (reward + discount * value_table[next_state]) 
                             for prob, next_state, reward, _ in env.P[state][action])
        if expected_value > best_value:
            best_value = expected_value
            best_action = action

    return best_value, best_action

def policy_iteration(env, discount=0.9, epsilon=1e-3):
    start = timer()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize policy randomly and value table as zeros
    policy = np.random.randint(n_actions, size=n_states)
    value_table = np.zeros(n_states)

    is_policy_stable = False
    while not is_policy_stable:
        value_table = policy_evaluation(env, policy, value_table, discount, epsilon)
        policy, is_policy_stable = policy_improvement(env, policy, value_table, discount)

    time_spent = timedelta(seconds=timer() - start)
    print(f"Solved in {np.count_nonzero(policy)} episodes and {time_spent} seconds")
    return policy, np.count_nonzero(policy), time_spent

def policy_evaluation(env, policy, value_table, discount, epsilon):
    """Evaluates the current policy."""
    while True:
        delta = 0
        for state in range(len(value_table)):
            old_value = value_table[state]
            value_table[state] = calculate_state_value(env, state, policy[state], value_table, discount)
            delta = max(delta, abs(old_value - value_table[state]))
        
        if delta < epsilon:
            break
    return value_table

def calculate_state_value(env, state, action, value_table, discount):
    """Calculates the value of a state under a given action."""
    return sum(prob * (reward + discount * value_table[next_state]) 
               for prob, next_state, reward, _ in env.P[state][action])

def policy_improvement(env, policy, value_table, discount):
    """Improves the policy based on evaluated values."""
    is_policy_stable = True
    for state in range(len(policy)):
        old_action = policy[state]
        policy[state] = np.argmax([calculate_action_value(env, state, action, value_table, discount) 
                                   for action in range(env.action_space.n)])
        if old_action != policy[state]:
            is_policy_stable = False
    return policy, is_policy_stable

def calculate_action_value(env, state, action, value_table, discount):
    """Calculates the value of an action for a given state."""
    return sum(prob * (reward + discount * value_table[next_state]) 
               for prob, next_state, reward, _ in env.P[state][action])

        

def train_and_test_pi_vi(env, discount=[0.9], epsilon=[1e-9], mute=False):
    vi_dict = run_iterations(env, discount, epsilon, value_iteration, "Value iteration", mute)
    pi_dict = run_iterations(env, discount, epsilon, policy_iteration, "Policy iteration", mute)
    return vi_dict, pi_dict

def run_iterations(env, discount, epsilon, iteration_func, name, mute):
    results = {}
    for dis in discount:
        results[dis] = {}
        for eps in epsilon:
            policy, solve_iter, solve_time = iteration_func(env, dis, eps)
            mean_reward, mean_eps, _, _ = test_policy(env, policy)    
            results[dis][eps] = {
                "mean_reward": mean_reward,
                "mean_eps": mean_eps,
                "iteration": solve_iter,
                "time_spent": solve_time,
                "policy": policy
            }
            if not mute:
                print(f"{name} for discount {dis} and epsilon {eps} is done")
                print(f"Iteration: {solve_iter} time: {solve_time}")
                print(f"Mean reward: {mean_reward} - mean eps: {mean_eps}")
    return results
def map_discretize(the_map):
    mapping = {"S": 0, "F": 0, "H": -1, "G": 1}
    return np.array([[mapping[loc] for loc in row] for row in the_map])



def see_policy(MAPS, map_size, policy):
    map_name = str(map_size)+"x"+str(map_size)
    data = map_discretize(MAPS[map_name])
    np_pol = policy_numpy(policy)
    plt.imshow(data, cmap='hot',interpolation="nearest")

    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            text = plt.text(j, i, arrow,
                           ha="center", va="center", color="yellow")
    plt.show()
def policy_numpy(policy):
    size = int(np.sqrt(len(policy)))
    return np.array(policy).reshape((size, size))


def plot_the_dict(dictionary, value="Score", size=4, variable="Discount Rate", log=False):
    plt.figure(figsize=(12, 8))
    title = f"Average and Max {value} on {size}x{size} Frozen Lake"

    data = []
    for k, v in dictionary.items():
        key_val = np.log10(k) if log else k
        for val in v:
            data.append({variable: key_val, value: float(val), "Type": "Average with std"})
        max_val = max(v)
        data.append({variable: key_val, value: float(max_val), "Type": "Max"})

    df = pd.DataFrame(data)
    sns.lineplot(x=variable, y=value, hue="Type", style="Type", markers=True, data=df).set(title=title)
def convert_dict_to_dict(the_dict):
    discount_rewards, discount_iterations, discount_times = {}, {}, {}
    epsilon_rewards, epsilon_iterations, epsilon_times = {}, {}, {}

    # Populate discount-based dictionaries
    for disc, eps_dict in the_dict.items():
        discount_rewards[disc], discount_iterations[disc], discount_times[disc] = [], [], []
        for eps, values in eps_dict.items():
            discount_rewards[disc].append(values['mean_reward'])
            discount_iterations[disc].append(values['iteration'])
            discount_times[disc].append(values['time_spent'].total_seconds())

    # Populate epsilon-based dictionaries
    for eps in the_dict[next(iter(the_dict))]:  # Using first key of the_dict for epsilon keys
        epsilon_rewards[eps], epsilon_iterations[eps], epsilon_times[eps] = [], [], []
        for disc, eps_dict in the_dict.items():
            epsilon_rewards[eps].append(eps_dict[eps]['mean_reward'])
            epsilon_iterations[eps].append(eps_dict[eps]['iteration'])
            epsilon_times[eps].append(eps_dict[eps]['time_spent'].total_seconds())

   


if __name__ == "__main__":
    print("Hello")
    mp = init_map()
    env = gym.make("FrozenLake-v0")
    dis = [0.4,0.5,0.6, 0.7,0.75,0.8,0.9, 0.95, 0.99, 0.9999]
    eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5,1e-6, 1e-7, 1e-8, 1e-9, 1e-10,1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
    vi_dict, pi_dict = train_and_test_pi_vi(env, discount=dis, 
                                            epsilon=eps, mute=True)
    vi_dict[0.99]
    pi_dict[0.9999]
    pol = vi_dict[0.99][1e-12]['policy']
    vi4 = convert_dict_to_dict(vi_dict)
    see_policy(mp, 4, pol)





