# python test_use_ppo.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO

from custom_env import CustomEnv

# 检测设备
device = torch.device("cpu")
print("device: ", device)

model = PPO.load("ppo_snake_custom_env", device=device)

env = CustomEnv(grid_size=10, max_steps=300, render_mode="human")
obs, _ = env.reset()

for _ in range(10000):
    env.render()
    # 模型预测动作
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()