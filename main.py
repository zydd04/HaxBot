import cv2
import numpy as np

ball_img = cv2.imread('data/ball.png', cv2.IMREAD_COLOR)
field_img = cv2.imread('data/field.png', cv2.IMREAD_COLOR)
ennemy_img = cv2.imread('data/ennemy.png', cv2.IMREAD_COLOR)
teamate_img = cv2.imread('data/team.png', cv2.IMREAD_COLOR)
t_post = cv2.imread('data/post-blue.png', cv2.IMREAD_COLOR)
e_post = cv2.imread('data/post-red.png', cv2.IMREAD_COLOR)
player = cv2.imread('data/player.png', cv2.IMREAD_COLOR)

obj_dict = {
    "Ball": {
        "img": ball_img,
        "threshold": 0.60,
        "color": (0, 255, 255)
    },
    "Ennemy": {
        "img": ennemy_img,
        "threshold": 0.60,
        "color": (255, 0, 0)
    },
    "Team": {
        "img": teamate_img,
        "threshold": 0.70,
        "color": (0, 0, 255)
    },
    "Team Post": {
            "img": t_post,
            "threshold": 0.90,
            "color": (0, 0, 255)
    },
    "Ennemy Post": {
            "img": e_post,
            "threshold": 0.90,
            "color": (255, 0, 0)
    },
    "player": {
        "img": player,
        "threshold":0.7,
        "color": (0, 0, 255)
    }
}

for obj, cfg in obj_dict.items():
    needle = cfg["img"]
    h, w = needle.shape[:2]
    box_color = cfg["color"]
    thresh = cfg["threshold"]
    result = cv2.matchTemplate(field_img, needle, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= thresh)

    boxes = []
    scores = []
    for pt in zip(*loc[::-1]):
        x, y = int(pt[0]), int(pt[1])
        boxes.append([x, y, w, h])
        scores.append(float(result[y, x]))

    if len(boxes) == 0:
        continue

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=thresh, nms_threshold=0.3)
    for i in indices:
        idx = i if isinstance(i, (list, np.ndarray)) else i
        x, y, box_w, box_h = boxes[idx]

        cv2.rectangle(field_img, (x, y), (x + box_w, y + box_h), box_color, 2)

        label_text = f"{obj} ({scores[idx]:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255) 

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        text_y = y - 6 if y - 6 > text_h else y + box_h + text_h + 6
        
        cv2.putText(field_img, label_text, (x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1800, 900)
cv2.imshow('Detection', field_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
