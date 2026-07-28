# python evaluate.py

import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_LunarLander")

# 批量评估
eval_episodes = 50
total_reward = 0
for ep in range(eval_episodes):
    obs, _ = env.reset()
    ep_r = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_r += reward
        if terminated or truncated:
            break
    total_reward += ep_r
    print(f"Episode {ep+1} reward: {ep_r:.2f}")
print(f"平均奖励：{total_reward/eval_episodes:.2f}")
env.close()
