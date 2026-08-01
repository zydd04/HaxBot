import cv2
import time

from capture import WindowCapture
from detector import Detector

WINDOW_TITLE = "HaxBall Play — Mozilla Firefox"


def main():

    try:
        capture = WindowCapture(window_title=WINDOW_TITLE, crop=(485, 260, 1450, 720))
    except RuntimeError as e:
        print(e)
        return
    detector = Detector()
    cv2.namedWindow("HaxBot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HaxBot", 1400, 800)
    last = time.time()
    try:
        while True:

            frame = capture.get_frame()

            if frame is None:
                continue

            state = detector.detect(frame)
            detector.draw(frame, state)
            fps = 1 / max(time.time() - last, 1e-6)
            last = time.time()
            cv2.putText(frame,f"FPS: {fps:.1f}",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),)
            cv2.putText(frame,f"Player: {state['player']}",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.putText(frame,f"Ball: {state['ball']}",(10, 70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.imshow("HaxBot", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()