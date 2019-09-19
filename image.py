import cv2
import argparse
import numpy as np
from face_detector import face_detector


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="", help="path to input image file")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
frame = img.copy()
image_o, bbox = face_detector(frame)

name = "out_" + args["image"]
cv2.imwrite(name,image_o)

count = 0
for ix in bbox:
	count += 1
	cropped_frame = img[ix[1]:ix[3],ix[0]:ix[2]]
	name = str(count) + args["image"].split("/")[-1]
	cv2.imwrite(name,cropped_frame)

