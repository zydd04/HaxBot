import numpy as np

class Reward():
    def __init__(self):
        self.prev_dist = None
        self.prev_goal_dist = None
    def reset():
        self.prev_dist = None
        self.prev_goal_dist = None
    def compute(self, player_pos, ball_pos, ennemy_post_pos, scored=False, conceded=False):
        reward = 0.0
        if scored: reward += 100.0
        if conceded: reward -= 100.0
        ball_dist = np.linalg.norm(np.array(player_pos) - np.array(ball_pos))
        if self.prev_dist is not None:
            if ball_dist < self.prev_dist:
                reward += 0.5
            else: reward -= 0.25
        self.prev_dist = ball_dist
        goal_dist = np.linalg.norm(mp.array(ball_pos) - np.array(ennemy_post_pos))
        if self.prev_goal_dist is not None:
            if goal_dist < self.prev_goal_dist:
                reward += 1.00
            else:
                reward -= 0.5

        self.prev_goal_dist = goal_dist
        if ball_dist < 20:
            reward += 1.50
        return reward


