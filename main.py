import cv2
import numpy as np
from keras.models import load_model
from src import PreProcessing, Ellipse
from configuration import *

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

## The model and the labels will be created by Teachable Machine
## where our AI will be created and trained
# Load the model
model = load_model("keras_Model.h5", compile=False)
# Load the labels
class_names = open("labels.txt", "r").readlines()

# Get the camera image
if CAMERA == 0:
    video = cv2.VideoCapture(0)
elif CAMERA == 1:
    video = cv2.VideoCapture(f"http://{CAMERA_IP}:{CAMERA_PORT}/video")

while True:
    # Get the image from the camera or a path pre-defined
    if CAMERA == 0 or CAMERA == 1:
        ret, image = video.read()
    elif CAMERA == 2:
        image = cv2.imread(IMAGE_INPUT_PATH)
    else:
        raise TypeError("Select the type of the camera")

    img_result =  image.copy()

    # Pre-processing the image
    imagePreProcessed = PreProcessing.ImagePreProcessing(image.copy())
    #cv2.imshow('Image Pre-processed', imagePreProcessed)

    # Recognize ellipses in the image
    ellipses = Ellipse.RecognizeEllipses(image.copy(), imagePreProcessed.copy())
    print(f'Number of ellipses recognized: {len(ellipses)}')

    # Find contours in the image
    # Parameters:
    #   - RETR_EXTERNAL: Returns only the external contours
    #   - CHAIN_APPROX_NONE: Stores all the contour points
    countors, _ = cv2.findContours(imagePreProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Process each ellipse
    for ellipse in ellipses:
        # Calculate the area of the ellipse
        area = cv2.contourArea(ellipse)

        # Check if the area of the ellipse is significant
        if area > 500:
            # Founding a rectangle around the ellipse
            x, y, w, h = cv2.boundingRect(ellipse)

            # Cut the image to show only the coin, it's important to improve the processing
            image_cutout = image.copy()[y:y + h, x:x + w]

            # Apply Gaussian blur to the 'image' to reduce noise and smooth the image
            # Parameters:
            #   - (3, 3): Kernel size for blurring, where (3, 3) represents a 3x3 pixel neighborhood
            #   - 3: Standard deviation of the Gaussian kernel, controlling the amount of smoothing
            image_cutout = cv2.GaussianBlur(image_cutout, (3, 3), 3)
            #cv2.imshow('Image cutout', image_cutout)

            ##############

            # Resize the raw image into (224-height,224-width) pixels
            image_cutout = cv2.resize(image_cutout, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow('Image cutout resized', image_cutout)

            # Make the image a numpy array and reshape it to the models input shape.
            image_cutout = np.asarray(image_cutout, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image_cutout = (image_cutout / 127.5) - 1

            # Predicts the model
            prediction = model.predict(image_cutout)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            classe = class_name[2:]

            conf = str(np.round(confidence_score * 100))[:-2]

            if int(conf) > 50:
                print("Class:", classe, end="")
                print("Confidence Score:", conf, "%")

                img_result = cv2.putText(img_result, f'{classe}: {conf}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Resultado", img_result)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # If keyboard the esc was pressed (27) or the type of the camera is 2, stop the loop
    if keyboard_input == 27 or CAMERA == 2:
        break

if CAMERA == 0 or CAMERA == 1:
    video.release()
elif CAMERA == 2:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    raise TypeError("Select the type of the camera")

# Close all windows
cv2.destroyAllWindows()
