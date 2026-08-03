# test_gymnasium.py

import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print('---')
    print("action: " + str(action))
    print("observation: " + str(observation))
    print("reward: " + str(reward))
    print("terminated: " + str(terminated))
    print("truncated: " + str(truncated))
    print("info: " + str(info))

    if terminated or truncated:
        observation, info = env.reset()

env.close()