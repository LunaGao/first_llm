from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from custom_env import CustomEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

script_dir = Path(__file__).resolve().parent
save_path = script_dir / "ppo_snake_custom_env_train3"
tensorboard_dir = script_dir / "ppo_logs"

env = CustomEnv(grid_size=10, max_steps=300)
check_env(env, warn=True)

# 这轮仍然保留 train2 之后的小步微调思路：
# 1. 进一步降低学习率，减少后期抖动。
# 2. 略微降低熵系数，让策略从探索转向收敛。
# 3. 收紧 clip_range，让 PPO 更新更保守。
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log=str(tensorboard_dir),
    learning_rate=5e-5,
    n_steps=512,
    batch_size=256,
    n_epochs=8,
    gamma=0.995,
    gae_lambda=0.98,
    ent_coef=0.005,
    clip_range=0.10,
    policy_kwargs={
        "net_arch": [256, 256],
    },
)

model.learn(
    total_timesteps=200_000,
    tb_log_name="test_03",
)

env.close()

# 保存模型
model.save(str(save_path))
print("saved:", save_path)
