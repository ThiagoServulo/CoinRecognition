# Coin Recognition with Python and Teachable Machine

This project uses Teachable Machine to recognize coins in a video stream from a camera or a pre saved image. 
It identifies the type of coin and overlays the coin image with the corresponding value label.

## Technical Details

These are some technical objectives that I wanted to learn and improve by constructing this system:

1. **Model Selection**: The heart of this project is the machine learning model used for coin recognition. This involves the following steps:

   * Data Collection: Gathering a dataset of coin images, covering that coin types that you need, in different orientations, and lighting conditions.

   * Data Preprocessing: Preprocessing the dataset by resizing images to a consistent resolution, normalizing pixel values, and augmenting the dataset with techniques like rotation, flipping, and brightness adjustments.

   * Model Architecture: Selecting a suitable deep learning architecture for image classification, in this case it will be used the Teachable Machine. 

   * Training: Training the chosen model using the preprocessed dataset. This involves forward and backward passes through the network to adjust model weights.


2. **Integration with OpenCV**: To interact with the camera and process video frames, it will be used the OpenCV library. Here's how it's integrated:

    * Camera Initialization: Initializing the camera stream using OpenCV. This involves selecting the camera source (webcam or camera IP) and setting parameters like frame resolution and frame rate.

    * Frame Capture: Continuously capturing video frames from the camera stream.

    * Preprocessing: Preprocessing each frame, if necessary, to match the format expected. And it's necessary resizing, normalizing pixel values, and potentially converting to grayscale.

    * Recognize the contours of ellipses: Since every coin has a circular shape, only the contours with circular shapes will be processed, to save processing. And, as the image may distort slightly due to lighting and video quality factors, the contours will not approximate perfect circles, but rather ellipses that resemble circles.


3. **Real-time inference**: The main challenge of this project is to achieve real-time inference, which means processing video frames at a speed that matches the camera's frame rate. To optimize performance:

    * Quantization: During the recognition step, quantization techniques can be applied to reduce model size and improve inference speed.

    * Minimizing interference: Using some parameters as a basis for configuring the software, we can minimize noise in the image and discard contours with shapes that we are not interested in, this increases the program's real-time response.

## How to use this program to create your own model

To use this program to create your own model, you need to follow these steps:

1. **Choosing the Camera Type**: To run this code, you need to create a configuration file (configurations.py) inside the "src" folder with the information about the camera you will use. The following example includes the basic information that should be in this file:

```python
# Camera type of connection
#   0 = webcam
#   1 = remote camera (using IP to connect)
#   2 = image saved
CAMERA = 1

## In the case CAMERA = 1
# Camera connection IP
CAMERA_IP = "XX.XX.XX.XX" # Insert your IP
# Camera connection port
CAMERA_PORT = "XXXX" # Insert your port

## In the case CAMERA = 2
IMAGE_INPUT_PATH = 'path/image.jpg' # Insert the path for your image
```

* For the camera type, you can use a webcam from your computer or the camera from your smartphone. However, if you choose the latter, you will need to download an app to connect the camera to the program. I recommend using the "DroidCam" app (Look for this app on your phone's app store). But you can use another app that allows you to connect the camera of your smartphone using an IP and port. Alternatively, you can use pre-saved images too.

![Droidcam][droidcam]

2. **Collecting samples**: To create your own model, you first need to select the coins that you will use in your model. Next, you need to take many different photos of your coin to use as samples for you model. These photos must have various angles, different illuminations, and from both sides. There isn't an exact number, but the more photos you take, the more accurate your model is likely to be. To assist you in this process, I have created a code (captureImagesToTraining.py) inside the "training" folder. This code captures a new photo of your coin whenever you press the 's' key on your keyboard. When you have taken enough photos, simply press 'q' to quit. This code will save the photos in a folder, crop the photo to only consider the coin, preprocess the image with a filter, and resize the image.

![Folder with coins][coins_to_training_1]

![Samples to training][coins_to_training_2]


3. **Building your model**: With the samples for different coins, you need to create your model. To do this, you can access the official site of [Teachable Machine], select the option "Image Project," choose "Standard image model", and upload your images, creating a new class for each type of coin. Next, click on "Training" and wait until the model is created. 

![Creating the model][creating_model]

* Finally, click on the option "Export model", and in the "TensorFlow" tab, select the option "Keras," then click on "Download my model." You will download a zip file containing the Keras model ("keras_model.h5") and the labels ("labels.txt"), which represent the names of your coins.

![Downloading the model][download_your_model]

![Model][model]

4. **Running the Program**: To execute the program, simply run the main file. The program will automatically connect to the selected camera, and you will see the image that the camera is pointing to. When the system recognizes a type of coin based on your model, this coin will be encircled with a green circle, the name of the coin will be printed above, and the precision of this recognition will also be displayed. The following image serves as an example of the input image captured by the camera and the output image with the coins recognized by this system.

![Input][input]

![Output][output]

## Possible improvements and recognition issues that you may encounter

One of the biggest challenges in dealing with image recognition is the influence of illumination. This occurs because, during image capture, factors such as contrast, shading, brightness, and reflections may directly interfere with the results. These issues can lead to non-recognition, meaning that the systems may fail to identify a coin in your image. Or the system may incorrectly recognize the coin, indicating the wrong type of coin.

To minimize these effects, you can adjust certain parameters in the filters applied to the image. In the file "PreProcessing.py," you will find all the applied filters along with explanations of each filter's parameters. Depending on your images, you may need to fine-tune these filters to reduce these effects and enhance recognition. Also, if you want to observe this filter being applied, simply uncomment the lines that display the windows. It may help you see the effects of your changes.

![Blur filter][blur]

![Border filter][border]

![Erode filter][erode]

Another possibility to improve this solution is by using different filters or constructing a system with standard illuminations, as variations in lighting can affect recognition accuracy. Additionally, taking more photos to create the model may result in a more accurate recognition.

## Versions

**V1.0.1**
* First version (18/11/2023)

## Author
- [@thiagoservulo](https://github.com/ThiagoServulo)

- 11/18/2023

[droidcam]: images/droidcam.png
[coins_to_training_1]: images/coins_to_training.png
[coins_to_training_2]: images/coins_to_training_2.png
[Teachable Machine]: https://teachablemachine.withgoogle.com/
[creating_model]: images/creating_model.png
[download_your_model]: images/download_your_model.png
[model]: images/model.png
[input]: images/input.png
[output]: images/output.png
[blur]: images/blur_filter.png
[border]: images/border_filter.png
[erode]: images/erode_filer.png