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

            # Resize the image
            # Parameters:
            #   - (224, 244): New image size in pixels
            #   - cv2.INTER_AREA: The average of the pixels is calculated within an original pixel area
            image_cutout = cv2.resize(image_cutout, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow('Image cutout resized', image_cutout)

            # Make the image a numpy array and reshape it to the models input shape
            # Parameters:
            #   - 1: indicates that we are dealing with a single sample
            #   - 224 indicates the width of the image after resizing
            #   - 224 indicates the height of the image after resizing
            #   - 3 indicates that the image has 3 color channels (RGB)
            image_cutout = np.asarray(image_cutout, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            # Dividing by the maximum possible value (127.5 in this case)
            # Next, the 1 is subtracted from each pixel value. This adjusts the values to the desired range of -1 to 1
            image_cutout = (image_cutout / 127.5) - 1

            # Make predictions using the model created
            prediction = model.predict(image_cutout)

            # Find the index of the maximum value in the prediction array
            index = np.argmax(prediction)

            # Determine the predicted class label associated with the input data
            class_name = class_names[index]

            # Get the score of the confidence recognizing
            confidence_score = prediction[0][index]

            # Get the class name (The name of the coing recognized)
            coin_name = class_name[2:-1]

            # Calculate the confidence score in percentage
            confidence_score = str(np.round(confidence_score * 100))[:-2]

            # Check the score of the confidence
            if int(confidence_score) > 50:
                print("Coin name:", coin_name)
                print("Confidence Score:", confidence_score, "%")

                # Write the text with the coin information in the image
                # Parameters:
                #   - (x, y): These are the coordinates where the text will be written
                #   - cv2.FONT_HERSHEY_SIMPLEX: The font type to be used for the text
                #   - 0.5: Defines the font scale factor, which determines the size of the text relative to the image
                #   - (255, 255, 255): The color of the text in the format (B, G, R)
                #   - 2: Specifies the thickness of the text
                image = cv2.putText(image.copy(), f'{coin_name}: {confidence_score}', (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                print(f"Coin desconsidered because your confidence score is too low")
                #print("Coin name:", coin_name)
                #print("Confidence Score:", confidence_score, "%")

        # Show the final image, with the coins recognized
        cv2.imshow("Coins recognized", image)

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
