import os
import numpy as np
from video import *

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(3,3),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

def captureImages(coinName: str, cleanFolder: bool = False) -> None:
    """
    Captures and saves images from a camera for a given coin name.

    :param coinName: The name of the coin for which to capture images.
    :param cleanFolder:  Whether to clean the folder before saving new images. Default is False.
    :return: None
    """
    # Starts video capture from a camera
    video = startCamera()
    # Set the folder where the images will be saved
    folder = f'imagesToTraining/{coinName}'
    # If the folder does not exist, create it
    if not os.path.exists(folder):
        os.mkdir(folder)

    # If it required, clean the folder before save the new images
    if cleanFolder:
        # Listing files in the folder
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            # Removing files
            if os.path.isfile(path):
                os.remove(path)

    # Initializing the image index
    indexImage = 700

    # Loop to capture and proccess the images
    while True:
        # Capturing image
        _,image = video.read()
        # Resizing the image
        image = cv2.resize(image,(640,480))
        # Pre processing the image
        imagePreProcessed = preProcess(image)
        # Find contours in the image
        countors,_ = cv2.findContours(imagePreProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Processing all countours found
        for cnt in countors:
            # Calculing the contour  area
            area = cv2.contourArea(cnt)
            # Check if the area is good
            # This step is important to ignore the noises in the image
            if area > 2000:
                # Founding a rectangle around the contour
                x, y, w, h = cv2.boundingRect(cnt)
                imageCropped = image[y: y + h, x: x + w]
                # Resizing the cropped image
                # This step is important because the Teachable Machine requires images with this size
                imageCropped = cv2.resize(imageCropped, (224, 224))
                # Show the cropped and resized image
                cv2.imshow('Image', image)
                # Check the keyboard keys interruptions
                key = cv2.waitKey(1) & 0xFF
                # If the key is 's' capture and save the image
                # If the key is 'q' close all windows and stop the program
                if key == ord('s'):
                    cv2.imwrite(f'{folder}/imagem{indexImage}.jpg', image)
                    indexImage += 1
                    print(f'Saving image number: {indexImage}')
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

captureImages('teste')