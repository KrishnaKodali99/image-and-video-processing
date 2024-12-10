import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test2.jpg')

if image is None:
    print("Could not read input image")
    exit()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image)

faces_image = []
for (x,y,w,h) in faces:
    crop_image = image[y:y+h, x:x+w]
    faces_image.append(crop_image)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0, 0, 255), 1)
    
for f in faces_image:
    cv2.imshow('img',f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

