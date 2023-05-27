import argparse
import cv2


cap = cv2.VideoCapture("clock.mp4")
success,frame = cap.read()

while success:
    

