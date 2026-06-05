import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery = False, render_mode = 'human')
q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))
rewards_per_episode = []
epsilon_decay_graph = []

learning_rate = 0.9 # alpha
discount_factor = 0.9 # gamma
epsilon = 0.9 # random actions
epsilon_decay_rate = 0.0001
n_episodes = 3000

for i in range(n_episodes):
    state = env.reset()[0] # Generates a new starting state

    terminated, truncated = False, False

    accumulated_reward = 0


    # Choose initial action using epsilon-greedy policy
    if (np.random.random() < epsilon): # Exploaratory action 90% of the time
            action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    while (not terminated and not truncated):
        
        new_state, reward, terminated, truncated, info = env.step(action)

        # Choose the NEXT action using the exact same epsilon-greedy policy
        if (np.random.random() < epsilon): # Exploaratory action 90% of the time
            new_action = env.action_space.sample()
        else:
            new_action = np.argmax(q_table[new_state])
        
        # Q - Learning
        # if terminated == True:
        #     q_table[state][action] += learning_rate * ((reward + discount_factor * 0) - q_table[state][action])
        # else: 
        #     q_table[state][action] += learning_rate * ((reward + discount_factor * np.max(q_table[new_state])) - q_table[state][action])

        # SARSA
        if terminated == True:
            q_table[state][action] += learning_rate * ((reward + discount_factor * 0) - q_table[state][action])
        else: 
            q_table[state][action] += learning_rate * (reward + (discount_factor * q_table[new_state][new_action]) - q_table[state][action])



        accumulated_reward += reward
        action = new_action
        state = new_state
    rewards_per_episode.append(accumulated_reward)
    epsilon = max(epsilon - epsilon_decay_rate, 0.1)
    epsilon_decay_graph.append(epsilon)
env.close()