import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from custom_env import CustomEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

env = CustomEnv(grid_size=10, max_steps=300)
check_env(env, warn=True)

# 以 train1 的结构为基准，只做训练参数优化。
# 第一轮结果已经有上升趋势，因此第二轮主要目标是：
# 1. 降低学习率，减少后期震荡。
# 2. 增加探索，避免策略过早收敛。
# 3. 拉长训练步数，让模型继续学习。
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log="./ppo_logs/",  # 日志存放目录
    learning_rate=1e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.98,
    ent_coef=0.01,
    clip_range=0.15,
    policy_kwargs={
        "net_arch": [256, 256],
    },
)
model.learn(
    total_timesteps=300_000,
    tb_log_name="test_02",
)

env.close()

# 保存模型
model.save("ppo_snake_custom_env_train2")
print("done")
