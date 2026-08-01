import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PlayerAction:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    SHOOT = 4
    N_ACTIONS = 5


class FieldObj:
    EMPTY = 0
    WALL = -1
    PLAYER = 2
    BALL = 1


class HaxBallEnv(gym.Env):

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        field_w: int = 62,
        field_h: int = 26,
        goal_size: int = 6,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.field_w = field_w
        self.field_h = field_h
        self.render_mode = render_mode

        self.goal_size = goal_size
        self.goal_top = (field_h - goal_size) // 2
        self.goal_bottom = self.goal_top + goal_size  # exclusive

        self.action_space = spaces.Discrete(PlayerAction.N_ACTIONS)
    
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(max(field_w, field_h)),
            shape=(4,),
            dtype=np.float32,
        )

        self.player_pos = None
        self.ball_pos = None

    def _in_goal_mouth(self, y: int) -> bool:

        return self.goal_top <= y < self.goal_bottom

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.player_pos = [self.field_w // 3, self.field_h // 2]
        self.ball_pos = [
            int(self.np_random.integers(1, self.field_w - 1)),
            int(self.np_random.integers(1, self.field_h - 1)),
        ]

        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        reward = -0.01
        terminated = False
        truncated = False

        if action == PlayerAction.LEFT and self.player_pos[0] > 0:
            self.player_pos[0] -= 1
        elif action == PlayerAction.RIGHT and self.player_pos[0] < self.field_w - 1:
            self.player_pos[0] += 1
        elif action == PlayerAction.UP and self.player_pos[1] > 0:
            self.player_pos[1] -= 1
        elif action == PlayerAction.DOWN and self.player_pos[1] < self.field_h - 1:
            self.player_pos[1] += 1
        elif action == PlayerAction.SHOOT:
            reward += self._try_shoot()
        dist_to_ball = abs(self.player_pos[0] - self.ball_pos[0]) + abs(
            self.player_pos[1] - self.ball_pos[1]
        )
        reward -= 0.001 * dist_to_ball

        scored_right = self.ball_pos[0] >= self.field_w - 1 and self._in_goal_mouth(self.ball_pos[1])
        scored_left = self.ball_pos[0] <= 0 and self._in_goal_mouth(self.ball_pos[1])
        if scored_right or scored_left:
            reward += 10.0
            terminated = True

        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _try_shoot(self) -> float:
        
        adjacent = (
            abs(self.player_pos[0] - self.ball_pos[0]) <= 1
            and abs(self.player_pos[1] - self.ball_pos[1]) <= 1
        )
        if not adjacent:
            return 0.0

        if self.player_pos[0] < self.ball_pos[0]:
            direction = 1
        elif self.player_pos[0] > self.ball_pos[0]:
            direction = -1
        else:
            direction = 1

        in_mouth = self._in_goal_mouth(self.ball_pos[1])
        if direction == 1:
            max_x = self.field_w - 1 if in_mouth else self.field_w - 2
            self.ball_pos[0] = min(self.ball_pos[0] + 10, max_x)
        else:
            min_x = 0 if in_mouth else 1
            self.ball_pos[0] = max(self.ball_pos[0] - 10, min_x)

        return 0.5

    def _get_obs(self):
        return np.array([*self.player_pos, *self.ball_pos], dtype=np.float32)

    def render(self):
        grid = [[FieldObj.EMPTY] * self.field_w for _ in range(self.field_h)]

        for x in range(self.field_w):
            grid[0][x] = FieldObj.WALL
            grid[self.field_h - 1][x] = FieldObj.WALL
        for y in range(self.field_h):
            if self._in_goal_mouth(y):
                continue  #goals
            grid[y][0] = FieldObj.WALL
            grid[y][self.field_w - 1] = FieldObj.WALL

        px, py = self.player_pos
        bx, by = self.ball_pos
        grid[py][px] = FieldObj.PLAYER
        grid[by][bx] = FieldObj.BALL

        symbols = {
            FieldObj.EMPTY: ".",
            FieldObj.WALL: "#",
            FieldObj.PLAYER: "P",
            FieldObj.BALL: "O",
        }
        text = "\n".join("".join(symbols[cell] for cell in row) for row in grid)

        if self.render_mode == "ansi":
            return text

        os.system("cls" if os.name == "nt" else "clear")
        print(text)
        print("P = player   O = ball   # = wall")
        return None


if __name__ == "__main__":
    env = HaxBallEnv(render_mode="human")
    obs, info = env.reset(seed=42)

    for step_num in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={step_num} action={action} reward={reward:.3f} obs={obs}")
        time.sleep(0.15)  
        if terminated or truncated:
            print("finished.")
            break