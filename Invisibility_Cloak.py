import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#time.sleep(3)
#background = 0

#for i in range(30):
ret, background = cap.read()

background = np.flip(background, axis=1)
#cv2.imshow('Display', background)
#cv2.waitKey(0)

while(cap.isOpened()):
    ret, img = cap.read()

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #value = (35, 35)
    #blured = cv2.GaussianBlur(hsv, value, 0)

    lower_red = np.array([0, 100, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 100, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    img[np.where(final == 0)] = background[np.where(final == 0)]

    cv2.imshow('Display', img)
    cv2.waitKey(1)