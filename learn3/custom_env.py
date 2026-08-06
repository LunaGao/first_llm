import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


# 方向常量：上、右、下、左
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3 


class CustomEnv(gym.Env):
    """一个适合强化学习训练的贪吃蛇环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, grid_size=10, max_steps=300, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        # 动作空间：
        # 0 = 继续直行
        # 1 = 左转
        # 2 = 右转
        self.action_space = spaces.Discrete(3)

        # 观测空间：3 个通道的网格
        # channel 0: 蛇头
        # channel 1: 蛇身
        # channel 2: 食物
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.uint8,
        )

        self.snake = []
        self.direction = RIGHT
        self.food = None
        self.steps = 0
        self.score = 0
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """重置环境并返回初始观测。"""
        super().reset(seed=seed)

        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center, center - 1),
            (center, center - 2),
        ]
        self.direction = RIGHT
        self.steps = 0
        self.score = 0
        self.food = self._spawn_food()

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        执行一步。

        返回：
        observation, reward, terminated, truncated, info
        """
        self.steps += 1
        self.direction = self._turn(self.direction, action)

        head_x, head_y = self.snake[0]
        move_x, move_y = self._direction_to_vector(self.direction)
        new_head = (head_x + move_x, head_y + move_y)

        reward = -0.01
        terminated = False
        truncated = False

        # 反向掉头（蛇头撞到自己身体的第二节）
        if len(self.snake) >= 2 and new_head == self.snake[1]:
            reward = -1.0
            terminated = True
        # 撞墙
        elif not self._in_bounds(new_head):
            reward = -1.0
            terminated = True
        # 咬到自己
        elif new_head in self.snake[:-1]:
            reward = -1.0
            terminated = True
        else:
            self.snake.insert(0, new_head)

            # 吃到食物
            if new_head == self.food:
                reward = 1.0
                self.score += 1
                if len(self.snake) == self.grid_size * self.grid_size:
                    terminated = True
                else:
                    self.food = self._spawn_food()
            else:
                self.snake.pop()

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """渲染环境。"""
        if self.render_mode == "human":
            self._render_frame()
            return None

        if self.render_mode == "rgb_array":
            return self._render_frame()

        return None

    def close(self):
        """释放资源。"""
        if pygame is not None:
            if self.window is not None:
                pygame.display.quit()
                self.window = None
            if self.clock is not None:
                self.clock = None
            pygame.quit()

    def _get_observation(self):
        """生成 3 通道网格观测。"""
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.uint8)

        if self.snake:
            head_x, head_y = self.snake[0]
            obs[0, head_x, head_y] = 1

        for x, y in self.snake[1:]:
            obs[1, x, y] = 1

        if self.food is not None:
            food_x, food_y = self.food
            obs[2, food_x, food_y] = 1

        return obs

    def _get_info(self):
        """返回调试和统计信息。"""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "food": self.food,
            "direction": self.direction,
        }

    def _spawn_food(self):
        """在空白位置生成食物。"""
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            return None

        random_index = int(self.np_random.integers(0, len(empty_cells)))
        return empty_cells[random_index]

    def _turn(self, current_direction, action):
        """根据动作更新朝向，禁止反向掉头。"""
        if action == 0:
            return current_direction
        if action == 1:
            new_direction = (current_direction - 1) % 4
        elif action == 2:
            new_direction = (current_direction + 1) % 4
        else:
            raise ValueError(f"非法动作: {action}")

        # 禁止反向掉头：新旧方向差为 2 时视为掉头（UP↔DOWN、LEFT↔RIGHT）
        if (new_direction - current_direction) % 4 == 2:
            return current_direction

        return new_direction

    def _direction_to_vector(self, direction):
        """将方向转换为位移向量。"""
        if direction == UP:
            return -1, 0
        if direction == RIGHT:
            return 0, 1
        if direction == DOWN:
            return 1, 0
        if direction == LEFT:
            return 0, -1
        raise ValueError(f"非法方向: {direction}")

    def _in_bounds(self, position):
        """检查坐标是否在地图内。"""
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _render_frame(self):
        """使用 pygame 渲染当前帧。"""
        if pygame is None:
            raise ImportError("未安装 pygame，无法进行图形化渲染，请先执行: pip install pygame")

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Snake CustomEnv")

            if self.clock is None:
                self.clock = pygame.time.Clock()

            canvas = self.window
        else:
            canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((25, 25, 25))

        # 先画网格，方便观察蛇和食物的位置。
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(canvas, (45, 45, 45), rect, width=1)

        # 蛇身使用深绿色，蛇头使用亮绿色。
        for row, col in self.snake[1:]:
            body_rect = pygame.Rect(
                col * self.cell_size + 2,
                row * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4,
            )
            pygame.draw.rect(canvas, (0, 160, 0), body_rect, border_radius=6)

        if self.snake:
            head_row, head_col = self.snake[0]
            head_rect = pygame.Rect(
                head_col * self.cell_size + 2,
                head_row * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4,
            )
            pygame.draw.rect(canvas, (0, 220, 0), head_rect, border_radius=8)

        # 食物使用红色圆形。
        if self.food is not None:
            food_row, food_col = self.food
            food_center = (
                food_col * self.cell_size + self.cell_size // 2,
                food_row * self.cell_size + self.cell_size // 2,
            )
            pygame.draw.circle(canvas, (230, 70, 70), food_center, self.cell_size // 3)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(canvas)
        return np.transpose(frame, (1, 0, 2))
