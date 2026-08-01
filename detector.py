import cv2
import numpy as np

GOAL_DIST_THRESHOLD = 30


class Detector:

    def __init__(self):

        self.templates = {
            "ball": {
                "img": cv2.imread("data/ball1.png"),
                "threshold": 0.70
            },
            "player": {
                "img": cv2.imread("data/player1.png"),
                "threshold": 0.70
            },
            "enemy_goal": {
                "img": cv2.imread("data/score_here.png"),
                "threshold": 0.95
            },
            "team_goal": {
                "img": cv2.imread("data/not_here.png"),
                "threshold": 0.95
            }
        }

        for name, entry in self.templates.items():
            if entry["img"] is None:
                print(f"[Detector] WARNING: template '{name}' failed to load "
                      f"from disk. Detection for '{name}' will always return None.")

        self.prev_ball = None

        self.last_confidences = {}
        self.last_state = {}

    def reset(self):
        # Call this at the start of each episode so ball_velocity doesn't
        # get computed against a ball position from the previous episode.
        self.prev_ball = None
        self.last_confidences = {}
        self.last_state = {}

    def match(self, frame, template, threshold):

        if template is None:
            return None, 0.0

        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            return None, max_val

        h, w = template.shape[:2]

        center = (
            max_loc[0] + w // 2,
            max_loc[1] + h // 2
        )

        return center, max_val

    def _dist(self, a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def detect(self, frame):

        state = {}
        confidences = {}

        for key in ("player", "ball", "enemy_goal", "team_goal"):
            pos, conf = self.match(
                frame,
                self.templates[key]["img"],
                self.templates[key]["threshold"]
            )
            state[key] = pos
            confidences[key] = conf

        self.last_confidences = confidences

        if state["ball"] is None:

            ball_velocity = (0.0, 0.0)

        elif self.prev_ball is None:

            ball_velocity = (0.0, 0.0)

        else:

            ball_velocity = (
                state["ball"][0] - self.prev_ball[0],
                state["ball"][1] - self.prev_ball[1]
            )

        self.prev_ball = state["ball"]

        state["ball_velocity"] = ball_velocity


        enemy_goal_dist = self._dist(state["ball"], state["enemy_goal"])
        team_goal_dist = self._dist(state["ball"], state["team_goal"])

        state["goal_scored"] = (
            enemy_goal_dist is not None and enemy_goal_dist < GOAL_DIST_THRESHOLD
        )
        state["goal_conceded"] = (
            team_goal_dist is not None and team_goal_dist < GOAL_DIST_THRESHOLD
        )

        self.last_state = state

        return state

    def draw(self, frame, state):

        colors = {
            "player": (0, 0, 255),
            "ball": (0, 255, 255),
            "enemy_goal": (255, 0, 0),
            "team_goal": (0, 0, 255)
        }

        for key in colors:

            pos = state.get(key)

            if pos is None:
                continue

            cv2.circle(frame, pos, 6, colors[key], -1)
            cv2.putText(
                frame,
                key,
                (pos[0] + 8, pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[key],
                1
            )

        if state.get("goal_scored"):
            cv2.putText(frame, "GOAL SCORED", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if state.get("goal_conceded"):
            cv2.putText(frame, "GOAL CONCEDED", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame