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
    "Team Post": {"img": t_post, "threshold": 1.00, "color": (0, 0, 255)},
    "Ennemy Post": {"img": e_post, "threshold": 1.00, "color": (255, 0, 0)},
    "player": {"img": player, "threshold": 0.70, "color": (0, 0, 255)},
}

def detect_objects(frame_bgr, obj_dict):
    """Run template matching for every entry in obj_dict against frame_bgr,
    draw a box + label for the single strongest match, and return the frame."""
    for obj, cfg in obj_dict.items():
        needle = cfg["img"]
        if needle is None:
            continue  # template failed to load, skip instead of crashing
 
        h, w = needle.shape[:2]
        box_color = cfg["color"]
        thresh = cfg["threshold"]
 
        result = cv2.matchTemplate(frame_bgr, needle, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
 
        if max_val < thresh:
            continue
 
        x, y = max_loc
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), box_color, 2)
 
        label_text = f"{obj} ({max_val:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255)
 
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        text_y = y - 6 if y - 6 > text_h else y + h + text_h + 6
 
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
                # win32 BitBlt already returns BGR, no color conversion needed
 
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
 
