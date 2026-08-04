from custom_env import CustomEnv


def main():
    """简单运行自定义环境，确认 reset 和 step 是否正常。"""
    env = CustomEnv(grid_size=6, max_steps=30, render_mode="human")

    observation, info = env.reset(seed=42)
    print("=== reset ===")
    print("observation.shape:", observation.shape)
    print("info:", info)
    env.render()

    for step_index in range(30):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"=== step {step_index + 1} ===")
        print("action:", action)
        print("observation.shape:", observation.shape)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("info:", info)
        env.render()

        if terminated or truncated:
            print("回合结束，重新 reset")
            observation, info = env.reset()
            print("observation.shape:", observation.shape)
            print("info:", info)
            env.render()

    env.close()


if __name__ == "__main__":
    main()
