import gymnasium as gym

env = gym.make('FrozenLake-v1', desc = ['SF','FG'], is_slippery = False, render_mode = 'human')

state = env.reset()[0]
terminated, truncated = False, False

print(f'Initial Starting State: {state}.')

while (not terminated and not truncated):
    action = env.action_space.sample()
    # 0: left
    # 1: down 
    # 2: right
    # 3: up
    new_state, reward, terminated, truncated, info =  env.step(action = action)
    state = new_state
    print(f'action {action}, new_state: {new_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}')
    
env.close()