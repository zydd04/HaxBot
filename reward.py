import numpy as np

DEADZONE_PX = 2.0

SMOOTHING = 0.3

class RewardFunction:
    def __init__(self):
        self.smoothed_dist = None
        self.smoothed_goal_dist = None

    def reset(self):
        self.smoothed_dist = None
        self.smoothed_goal_dist = None

    def compute(self, player_pos, ball_pos, enemy_goal_pos, scored=False, conceded=False):
        reward = 0.0

        if scored:
            reward += 100.0
        if conceded:
            reward -= 100.0

        if player_pos is not None and ball_pos is not None:
            ball_dist = float(np.linalg.norm(np.array(player_pos) - np.array(ball_pos)))

            if self.smoothed_dist is None:
                self.smoothed_dist = ball_dist
            else:
                prev = self.smoothed_dist
                self.smoothed_dist = SMOOTHING * ball_dist + (1 - SMOOTHING) * prev

                if self.smoothed_dist < prev - DEADZONE_PX:
                    reward += 0.5
                elif self.smoothed_dist > prev + DEADZONE_PX:
                    reward -= 0.25

            if ball_dist < 20:
                reward += 1.50

        if ball_pos is not None and enemy_goal_pos is not None:
            goal_dist = float(np.linalg.norm(np.array(ball_pos) - np.array(enemy_goal_pos)))

            if self.smoothed_goal_dist is None:
                self.smoothed_goal_dist = goal_dist
            else:
                prev_g = self.smoothed_goal_dist
                self.smoothed_goal_dist = SMOOTHING * goal_dist + (1 - SMOOTHING) * prev_g

                if self.smoothed_goal_dist < prev_g - DEADZONE_PX:
                    reward += 1.00
                elif self.smoothed_goal_dist > prev_g + DEADZONE_PX:
                    reward -= 0.5
        return reward