import numpy as np
import cv2
import time
import os
import sys
import psutil
import logging

def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.get_open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        print(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)
# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error opening video0, trying video1")
    #exit()
    cap = cv2.VideoCapture(1)
    time.sleep(3)
    if not cap.isOpened():
        print("error opening video1, trying -1")
        cap = cv2.VideoCapture(-1)
        time.sleep(3)
        if not cap.isOpened():
            print("no cameras available")
            exit()
# Set auto exposure to manual mode
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# Set exposure value (adjust as needed)
#exposure_value = 1  # Decrease for lower exposure
#cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Capture a frame
ret, frame = cap.read()
import random
random_number = str(random.randint(1,10000))
# Check if frame is read correctly
time.sleep(1.5)
if ret:
    # Save the frame as an image
    cv2.imwrite("./calibration-images/captured_image"+random_number+".jpg", frame)
    print("Image captured and saved successfully.")
else:
    print("Error capturing frame.")

# Release the camera
img = cv2.imread("./calibration-images/captured_image"+random_number+".jpg")
cap.release()
cv2.imshow('img',img)
cv2.waitKey(-1)

cv2.destroyAllWindows()
restart_program()
