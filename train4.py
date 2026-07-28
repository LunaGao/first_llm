# train4.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: ", device)

# 创建环境
env = gym.make("LunarLander-v3")

# 简单PPO测试训练
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    device=device, 
    tensorboard_log="./ppo_logs/",  # 日志存放目录
    # 参数调整
    learning_rate=LinearSchedule(start=1.0, end=0.05, end_fraction=0.2),  # 线性衰减学习率，解决后期震荡
    clip_range=0.17,                                  # 缩小clip，抑制approx_kl持续上涨
    vf_coef=0.8,                                      # 提高价值loss权重，改善explained_variance
    ent_coef=0.01,                                    # 增加基础探索，防止策略过早坍缩
    gae_lambda=0.96,                                  # 略微提升GAE，优势估计更平滑
    n_epochs=8,                                       # 减少迭代轮次，避免单批次过度更新
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    max_grad_norm=0.5,
    )
model.learn(
    total_timesteps=100_000,
    tb_log_name="test_01",    # 本次实验名称
    )

# 测试运行
obs, _ = env.reset()
for _ in range(100_000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

# 保存模型
model.save("ppo_LunarLander")
print("done")