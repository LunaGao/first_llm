# train7.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule

# 检测设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: ", device)

# 创建环境
env = gym.make("LunarLander-v3")

# 网络扩容，提升价值网络拟合能力
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
)

# 简单PPO测试训练模型配置
model = PPO(
    "MlpPolicy", 
    env,
    policy_kwargs=policy_kwargs,
    verbose=1, 
    device=device, 
    tensorboard_log="./ppo_logs/",  # 日志存放目录
    # 参数调整
    learning_rate=LinearSchedule(start=3e-4, end=1e-4, end_fraction=0.2),  # 线性衰减学习率，解决后期震荡
    clip_range=0.18,                                  # 小幅放开更新区间，帮助突破当前奖励平台
    vf_coef=0.95,                                        # 小幅继续提高价值损失权重，进一步强化价值拟合
    ent_coef=0.02,                                   # 提高熵系数，保留更多探索，防止提前锁死次优策略
    gae_lambda=0.97,                                  # 进一步平滑GAE优势估计
    n_epochs=6,                                      # 继续降低迭代次数，减轻单批次过度更新，缓解开局震荡
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    max_grad_norm=0.5,
    clip_range_vf=0.3,                               # 新增：约束价值网络更新幅度，抑制value_loss剧烈波动
    )
model.learn(
    total_timesteps=150_000,
    tb_log_name="test_01",    # 本次实验名称
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