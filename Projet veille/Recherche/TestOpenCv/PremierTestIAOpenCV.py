from turtle import width
import cv2
import numpy as np


cap = cv2.VideoCapture('Recherche\\TestOpenCv\\assets\\videoTest.mp4')


while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('a'):
        break
    
cap.release()
cv2.destroyAllWindows()
