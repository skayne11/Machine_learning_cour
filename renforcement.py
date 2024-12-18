import gymnasium as gym
import numpy as np
import random
env = gym.make('LunarLander-v2', render_mode='human')

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000
max_steps = 1000

state_bins = [np.linspace(-1.0, 1.0, 10) for _ in range(env.observation_space.shape[0])]
n_bins = tuple(len(bins) + 1 for bins in state_bins)
q_table = np.zeros(n_bins + (env.action_space.n,))

def discretize_state(state):
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

for episode in range(episodes):
    state, _ = env.reset(seed=42)
    state = discretize_state(state)
    total_reward = 0

    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)

        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value

        state = next_state

        total_reward += reward

        if terminated or truncated:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
