import cv2
import numpy as np

def ImagePreProcessing(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an input image by applying Gaussian blur, Canny edge detection, dilation, and erosion.

    :param image: The input image as a NumPy array.
    :return: The preprocessed image as a NumPy array.
    """
    # Apply Gaussian blur to the 'image' to reduce noise and smooth the image
    # Parameters:
    #   - (5, 5): Kernel size for blurring, where (5, 5) represents a 5x5 pixel neighborhood
    #   - 3: Standard deviation of the Gaussian kernel, controlling the amount of smoothing
    imageBlurred = cv2.GaussianBlur(image, (5, 5), 3)
    #cv2.imshow('Image Blurred', imageBlurred)

    # Apply the Canny edge detection algorithm to the 'imageBlurred' to detect edges
    # Parameters:
    #   - 90: Lower threshold for edge detection
    #         Edges with gradient magnitude below this value are discarded
    #   - 140: Upper threshold for edge detection
    #          Edges with gradient magnitude above this value are considered strong edges
    imageBorders = cv2.Canny(imageBlurred, 90, 140)
    #cv2.imshow('Image Borders', imageBorders)

    # Create a 4x4 square-shaped kernel using NumPy
    # Parameters:
    #   - (4, 4): The shape of the kernel, creating a 4x4 matrix
    #   - np.uint8: Data type of the kernel elements, representing unsigned 8-bit integers
    kernel = np.ones((4, 4), np.uint8)

    # Apply dilation to the 'imageBorders' to expand and thicken detected edges
    # Parameters:
    #   - kernel: The structuring element used for dilation, which defines the neighborhood for the operation
    #   - iterations=2: The number of times dilation is applied, controlling the extent of thickening
    imageDilated = cv2.dilate(imageBorders, kernel, iterations=2)
    #cv2.imshow('Image Dilated', imageDilated)

    # Apply erosion to the 'imageDilated' to shrink and refine the expanded edges
    # Parameters:
    #   - kernel: The structuring element used for erosion, defining the neighborhood for the operation
    #   - iterations=1: The number of times erosion is applied, controlling the extent of refinement
    imageEroded = cv2.erode(imageDilated, kernel, iterations=1)
    #cv2.imshow('Image Eroded', imageEroded)

    # Show the image pre-processed
    imagePreProcessed = imageEroded.copy()
    #cv2.imshow('Image Pre-Processed', imagePreProcessed)

    return imagePreProcessed