import cv2
import numpy as np
from typing import List

def biggestDifference(vector: List[float]) -> float:
    """
    Calculate the largest absolute difference between any two elements in the given vector.

    :param vector:  A list containing 4 elements.
    :return: The largest absolute difference between any two elements in the vector.
    :raises: TypeError: If the input vector does not contain exactly 4 elements.
    """
    # Checking the number of elements in the vector
    if len(vector) != 4:
        raise TypeError("The vector should contain 4 elements")

    # Initializing the variable to store the largest difference
    biggest_difference = abs(vector[0] - vector[1])

    # Comparing all combinations of pairs of elements in the vector
    for i in range(len(vector)):
        for j in range(i + 1, len(vector)):
            difference = abs(vector[i] - vector[j])
            if difference > biggest_difference:
                biggest_difference = difference

    return biggest_difference


def RecognizeEllipses(imageOriginal: np.ndarray, imagePreProcessed: np.ndarray) -> List[np.ndarray]:
    """
    Detects and recognizes ellipses in a preprocessed image.

    :param imageOriginal: The original image as a NumPy array.
    :param imagePreProcessed: The preprocessed image as a NumPy array.
    :return: A list of detected ellipses (contours) as NumPy arrays.
    """
    # Find contours in the preprocessed image using the external retrieval mode and a simple approximation
    # method for chain-coding
    contours, _ = cv2.findContours(imagePreProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store detected ellipses
    ellipses_detected = []

    # Process the contours list
    for contour in contours:
        if len(contour) >= 5:
            # Fit an ellipse to the current contour
            ellipse = cv2.fitEllipse(contour)

            # Obtain the vertices of the bounding box of the fitted ellipse and convert them to integers
            vertices = cv2.boxPoints(ellipse).astype(int)

            # Initialize the list of distance between vertices
            distance_between_vertices = []

            # Calculate distances between adjacent vertices and add them to the distance list
            for i in range(4):
                # Get the next vertice
                j = (i + 1) % 4

                # Calculate the distance
                distance = np.sqrt(((vertices[i] - vertices[j]) ** 2).sum())

                # Append the value
                distance_between_vertices.append(distance)

            # Convert list of distances to a NumPy vector
            vector_of_distances = np.array(distance_between_vertices)

            # Calculate the biggest difference between the vertices
            difference = biggestDifference(vector_of_distances)

            # In this case, it will be use a factor of 10% to aproximate the ellipse to a circle
            if (max(vector_of_distances) * 0.1) < difference:
                # The ellipse will be ignored
                #print('Ellipse ignored')
                pass
            else:
                # The ellipse will be considered
                # Append the ellipse
                ellipses_detected.append(contour)

                # Draw a green ellipse in the image
                cv2.ellipse(imageOriginal, ellipse, (0, 255, 0), 2)

                # Draw red dots on the vertices
                for point in vertices:
                    cv2.circle(imageOriginal, tuple(point), 3, (0, 0, 255), -1)

    # Show the image with the detected ellipses
    #cv2.imshow('Ellipses detected', imageOriginal)

    return ellipses_detected
