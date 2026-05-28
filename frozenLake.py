import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery = False, render_mode = 'human')
q_table = np.zeros(shape = (env.observation_space.n, env.action_space.n)) # init a 64 x 4 2d Array

learning_rate = 0.9 # alpha
discount_factor = 0.9 # gamma
epsilon = 0.9 # random actions
epsilon_decay_rate = 0.0001
n_episodes = 3

reward_per_episode = np.zeros(n_episodes)

for i in range(n_episodes):
    print(f'Episode {i + 1}')

    state = env.reset()[0]
    terminated, truncated = False, False

    reward_in_the_episode = 0

    while (not terminated and not truncated):
        # Policy
        if np.random.random() < epsilon: # Exploration
            action = env.action_space.sample()
        else: # Exploitation
            action = np.argmax(q_table[state])
        # 0: left
        # 1: down 
        # 2: right
        # 3: up
        new_state, reward, terminated, truncated, info =  env.step(action = action)
        reward_in_the_episode += reward
        
        q_table[state][action] += learning_rate * (reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action])

        state = new_state
        print(f'action: {action}, new_state: {new_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}')

    reward_per_episode[i] = reward_in_the_episode
    epsilon = max(epsilon - epsilon_decay_rate, 0) # decay epsilon until 0

    if (epsilon == 0):
        learning_rate = 0.0001

    if (reward == 1): # At the end of the episode, if we finish - reward = 1
        reward_per_episode[i] = 1

env.close()
sum_rewards = np.zeros(n_episodes)
for t in range(n_episodes):
    sum_rewards[t] = np.sum(reward_per_episode[max(0,t-100):(t+1)])

plt.plot(sum_rewards)
# plt.savefig('frozen_lake4x4.png')


f = open('frozen_lake4x4.pk1', 'wb')
pickle.dump(q_table,f) # Saving the Q_Table
f.close()


# view the q_table normally
with open('frozen_lake4x4.pk1', 'rb') as f:
    loaded_q_table = pickle.load(f)
print(loaded_q_table)