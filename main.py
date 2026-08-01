import cv2
import numpy as np
from time import time
from capture import WindowCapture

ball_img = cv2.imread('data/ball1.png', cv2.IMREAD_COLOR)
#ennemy_img = cv2.imread('data/ennemy.png', cv2.IMREAD_COLOR)
#teamate_img = cv2.imread('data/team.png', cv2.IMREAD_COLOR)
t_post = cv2.imread('data/not_here.png', cv2.IMREAD_COLOR)
e_post = cv2.imread('data/score_here.png', cv2.IMREAD_COLOR)
player = cv2.imread('data/player1.png', cv2.IMREAD_COLOR)

obj_dict = {
    "Ball": {"img": ball_img, "threshold": 0.70, "color": (255, 255, 255)},
    "Team Post": {"img": t_post, "threshold": 0.90, "color": (0, 0, 255)},
    "Ennemy Post": {"img": e_post, "threshold": 0.90, "color": (255, 0, 0)},
    "player": {"img": player, "threshold": 0.70, "color": (0, 0, 255)},
}


def detect_objects(frame_bgr, obj_dict):
    """Run template matching for every entry in obj_dict against frame_bgr,
    draw boxes + labels directly on frame_bgr, and return it."""
    for obj, cfg in obj_dict.items():
        needle = cfg["img"]
        if needle is None:
            continue  
 
        h, w = needle.shape[:2]
        box_color = cfg["color"]
        thresh = cfg["threshold"]
 
        result = cv2.matchTemplate(frame_bgr, needle, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= thresh)
 
        boxes, scores = [], []
        for pt in zip(*loc[::-1]):
            x, y = int(pt[0]), int(pt[1])
            boxes.append([x, y, w, h])
            scores.append(float(result[y, x]))
 
        if not boxes:
            continue
 
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=thresh, nms_threshold=0.3)
        for i in indices:
            idx = int(i[0]) if isinstance(i, (list, np.ndarray)) else int(i)
            x, y, box_w, box_h = boxes[idx]
 
            cv2.rectangle(frame_bgr, (x, y), (x + box_w, y + box_h), box_color, 2)
 
            label_text = f"{obj} ({scores[idx]:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_color = (255, 255, 255)
 
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_y = y - 6 if y - 6 > text_h else y + box_h + text_h + 6
 
            cv2.putText(frame_bgr, label_text, (x, text_y), font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)
 
    return frame_bgr
 
 
def main():
    target_window = "HaxBall Play — Mozilla Firefox"
    try:
        capture = WindowCapture(target_window)
    except RuntimeError as e:
        print(e)
        return
 
    loopt = time()
 
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1800, 900)
 
    try:
        while True:
            frame_bgr = capture.get_frame()
            if frame_bgr is not None:
 
                frame_bgr = detect_objects(frame_bgr, obj_dict)
 
                cv2.imshow('Detection', frame_bgr)
 
                dt = time() - loopt
                if dt > 0:
                    print(f"capturing at {1 / dt:.1f} FPS")
                loopt = time()
 
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
    finally:
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
