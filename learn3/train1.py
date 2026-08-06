from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from custom_env import CustomEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

env = CustomEnv(grid_size=10, max_steps=300)
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log="./ppo_logs/",  # 日志存放目录
)
model.learn(
    total_timesteps=100_000, 
    tb_log_name="test_01",
    )

env.close()

# 保存模型
model.save("ppo_snake_custom_env")
print("done")

