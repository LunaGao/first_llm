# python test_use_ppo.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: ", device)

model = PPO.load("ppo_LunarLander", device=device)

env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()

for _ in range(10000):
    # 模型预测动作
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()