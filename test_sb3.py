# test_sb3.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO

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
    tensorboard_log="./ppo_logs/"  # 日志存放目录
    )
model.learn(
    total_timesteps=10000,
    tb_log_name="test_01",    # 本次实验名称
    )

# 测试运行
obs, _ = env.reset()
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

# 保存模型
model.save("ppo_LunarLander")
print("done")