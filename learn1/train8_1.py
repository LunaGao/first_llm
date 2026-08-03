# python train8_1.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: ", device)

# 创建环境
env = gym.make("LunarLander-v3")

# 加载模型继续训练
model = PPO.load("ppo_LunarLander", env=env)
model.learn(
    total_timesteps=50_000,
    tb_log_name="test_01_8_1",
    reset_num_timesteps=False,
)

# 测试运行
obs, _ = env.reset()
for _ in range(20000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

# 保存模型
model.save("ppo_LunarLander")
print("done")