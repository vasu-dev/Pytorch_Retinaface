import cv2
import argparse
import numpy as np
from face_detector import face_detector


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output-game-u-{}.avi'.format(str(args["video"]).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

while(1):
    hasFrame, frame = cap.read()
    if not hasFrame:
        continue
    
    outOpencvDnn = face_detector(frame)
    cv2.imshow("GAME", outOpencvDnn)

    vid_writer.write(outOpencvDnn)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vid_writer.release()