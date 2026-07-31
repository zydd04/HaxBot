import cv2
import numpy as np
from time import time
import dxcam

camera = dxcam.create()
frame = camera.grab()  

camera.start(target_fps=60)  
loopt = time()

while True:
    frame = camera.get_latest_frame()
    if frame is not None:
        cv2.imshow("Video Capture", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"capturing at {format(1 / (time() - loopt))} FPS")
        loopt = time()
    if cv2.waitKey(1) == 27:
        camera.stop()
        cv2.destroyAllWindows()
        break

"""
Using ImageGrab (Worse FPS):

loopt = time()
while(True):
    ss = ImageGrab.grab() #take screeshots

    cv_ss = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR) #convert rgb 2 bgr
    cv2.imshow('Game Recording', cv_ss) 
    print(f"capturing at {format(1 / (time() - loopt))} FPS")
    loopt = time()
    if cv2.waitKey(1) == ord('\x1b'): #press esc to quit
        cv2.destroyAllWindows()
        break
"""
