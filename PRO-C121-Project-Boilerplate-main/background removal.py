# import cv2 to capture videofeed
from email.mime import image
import cv2

import numpy as np


# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480
cv2.resize(image,(640,480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])
        upper_bound = np.array([255,255,255])

        # thresholding image
        ret, thresh = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Threshold', thresh)

        kernel = np.ones((5,5), np.uint8)

        # inverting the mask
        Mask = cv2.inRange(mountain, lower_bound, upper_bound) 
        Mask = cv2.erode(Mask, kernel, iterations=1) 
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel) 
        Mask = cv2.dilate(Mask, kernel, iterations=1) 

        Mask = cv2.bitwise_not(Mask)

        # bitwise and operation to extract foreground / person
        backgroundModel = np.zeros((1,65), np.float64)
        foregroundModel = np.zeros((1,65), np.float64)
        rectangle = (20,100,150,150)

        cv2.grabCut(image, Mask, rectangle, backgroundModel, foregroundModel,3, cv2.GC_INIT_WITH_RECT)

        # final image
        final_image = np.where(frame == 0, frame, frame)

        # show it
        cv2.imshow('Mask' , Mask)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
