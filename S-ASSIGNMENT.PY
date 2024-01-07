import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the image
image_path = "BLACKGRAM-lower.png"
image = cv.imread(image_path)

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Display the original grayscale image
cv.imshow('Original Image', gray_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Calculate and plot the histogram of pixel values
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(histogram)
plt.title("Histogram for BLACKGRAM data")
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Define a simple transformation function
def simple_transformation(img):
    return np.clip(2.8 * (img - 20), 0, 255).astype(np.uint8)

# Apply the simple transformation to the image
transformed_image = simple_transformation(gray_image)

# Display the transformed image
cv.imshow('Transformed Image', transformed_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Apply Gaussian blur to the transformed image
blurred_image = cv.GaussianBlur(transformed_image, (9, 9), 7)

# Display the blurred image
cv.imshow('Blurred Image', blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Binarize the blurred image using a threshold
_, binary_mask = cv.threshold(blurred_image, 105, 255, cv.THRESH_BINARY)

# Apply the binary mask to the original colored image
result_image = cv.bitwise_and(image, image, mask=binary_mask)

# Display the original and resultant images
cv.imshow('Original Image', image)
cv.imshow('Resultant Image', result_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the resultant image
cv.imwrite("Result_Image.png", result_image)
