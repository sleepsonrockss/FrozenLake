import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery = False, render_mode = 'rgb_array')
q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))
rewards_per_episode = []
epsilon_decay_graph = []

learning_rate = 0.9 # alpha
discount_factor = 0.9 # gamma
epsilon = 0.9 # random actions
epsilon_decay_rate = 0.001
n_episodes = 10

# Monte Carlo
for i in range(n_episodes):

    print(f'Episode {i+1}')
    terminated, truncated = False, False
    states, rewards, actions = [], [], []
    state, info = env.reset()

    
    # The while loop below get the sequence of States in the episode
    while (not terminated and not truncated):

        states.append(state)
        
        if (np.random.random() < epsilon): # Exploratory action 90% of the time
                action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        actions.append(action)
        state = next_state
    
    T = len(states)
    print(f'Length of Episode {i+1}: {T}')
    returns = []

    for j in range(T):
        G_t = 0
        for k in range(j,T):
            G_t += (discount_factor ** (k - j)) * rewards[k]
        returns.append(G_t)
        print(f'Return for Step {j+1} which is in state {states[j]} is {G_t}')
    
    # Updating the q_values for use in the next episode
    for t in range(T):
        s_t = states[t]
        a_t = actions[t]
        G_t = returns[t]
        
        q_table[s_t][a_t] += learning_rate * (G_t - q_table[s_t][a_t])
    epsilon = max(0.1, epsilon - epsilon_decay_rate)


# f = open('frozen_lake4x4.pk1', 'wb')
# pickle.dump(q_table,f) # Saving the Q_Table
# f.close()


# # view the returns list normally
# with open('frozen_lake4x4.pk1', 'rb') as f:
#     loaded_q_table = pickle.load(f)
# print(loaded_q_table)