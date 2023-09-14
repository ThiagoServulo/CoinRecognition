import cv2
from typing import Optional
from configuration import *

def startCamera() -> Optional[cv2.VideoCapture]:
    """
    Starts video capture from a camera.

    :return: cv2.VideoCapture or None: A video capture object
             if the camera is started successfully, otherwise None.
    """
    if CAMERA == 0:
        return cv2.VideoCapture(0)
    elif CAMERA == 1:
        return cv2.VideoCapture(f"http://{CAMERA_IP}:{CAMERA_PORT}/video")
    else:
        return None