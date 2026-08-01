import time
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from capture import WindowCapture
from controls import perform_action, reset_controls, ACTIONS
from reward import RewardFunction
from detector import Detector

MAX_BALL_SPEED_PX = 40.0

VECTOR_DIM = 11


class HaxballEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super().__init__()

        self.capture = WindowCapture("HaxBall Play — Mozilla Firefox")
        self.detector = Detector()
        self.reward_fn = RewardFunction()

        self.action_space = spaces.Discrete(len(ACTIONS))


        x0, y0, x1, y1 = self.capture.crop
        self.crop_w = max(1, x1 - x0)
        self.crop_h = max(1, y1 - y0)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
            ),
            "vector": spaces.Box(
                low=-1.0, high=1.0, shape=(VECTOR_DIM,), dtype=np.float32
            ),
        })

        self.frame_skip = 2
        self.max_steps = 5000
        self.current_step = 0

    def preprocess(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            (84, 84),
            interpolation=cv2.INTER_AREA
        )

        gray = gray[..., np.newaxis]

        return gray.astype(np.uint8)

    def _norm_pos(self, pos):
        """Normalize a (x, y) pixel position to [0, 1] using crop dims.
        Returns (0.0, 0.0) if pos is None -- callers must also check the
        corresponding presence flag to distinguish 'missing' from 'at
        the origin'."""
        if pos is None:
            return 0.0, 0.0
        x = float(np.clip(pos[0] / self.crop_w, 0.0, 1.0))
        y = float(np.clip(pos[1] / self.crop_h, 0.0, 1.0))
        return x, y

    def _extract_features(self, state):
        """Build the fixed-length numeric feature vector from a detector
        state dict. Order must stay in sync with VECTOR_DIM."""

        player_x, player_y = self._norm_pos(state.get("player"))
        player_present = 1.0 if state.get("player") is not None else 0.0

        ball_x, ball_y = self._norm_pos(state.get("ball"))
        ball_present = 1.0 if state.get("ball") is not None else 0.0

        vx, vy = state.get("ball_velocity", (0.0, 0.0))
        vx = float(np.clip(vx / MAX_BALL_SPEED_PX, -1.0, 1.0))
        vy = float(np.clip(vy / MAX_BALL_SPEED_PX, -1.0, 1.0))

        goal_x, goal_y = self._norm_pos(state.get("enemy_goal"))
        goal_present = 1.0 if state.get("enemy_goal") is not None else 0.0

        features = np.array([
            player_x, player_y, player_present,
            ball_x, ball_y, ball_present,
            vx, vy,
            goal_x, goal_y, goal_present,
        ], dtype=np.float32)

        assert features.shape[0] == VECTOR_DIM
        return features

    def _build_observation(self, frame):
        """Run detection once and construct the full Dict observation.
        Returns (obs, state) so callers (step/reset) can reuse `state`
        for reward computation without detecting twice."""

        if frame is None:
            obs = {
                "image": np.zeros((84, 84, 1), dtype=np.uint8),
                "vector": np.zeros((VECTOR_DIM,), dtype=np.float32),
            }
            return obs, None

        state = self.detector.detect(frame)

        obs = {
            "image": self.preprocess(frame),
            "vector": self._extract_features(state),
        }

        return obs, state

    def get_observation(self):
        frame = self.capture.get_frame()
        obs, _ = self._build_observation(frame)
        return obs

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        reset_controls()

        self.reward_fn.reset()
        self.detector.reset()

        self.current_step = 0

        time.sleep(1)

        obs = self.get_observation()

        return obs, {}

    def step(self, action):

        self.current_step += 1

        for _ in range(self.frame_skip):

            perform_action(int(action))

            time.sleep(1 / 60)

        frame = self.capture.get_frame()

        obs, state = self._build_observation(frame)

        if state is None:
            reward = -1
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, {}

        reward = self.reward_fn.compute(
            player_pos=state["player"],
            ball_pos=state["ball"],
            enemy_goal_pos=state["enemy_goal"],
            scored=state["goal_scored"],
            conceded=state["goal_conceded"]
        )

        terminated = (
            state["goal_scored"] or
            state["goal_conceded"]
        )

        truncated = self.current_step >= self.max_steps

        info = {
            "state": state,
            "reward": reward
        }

        return obs, reward, terminated, truncated, info

    def render(self):

        frame = self.capture.get_frame()

        if frame is not None:
            cv2.imshow("HaxBall AI", frame)
            cv2.waitKey(1)

    def close(self):

        reset_controls()

        cv2.destroyAllWindows()