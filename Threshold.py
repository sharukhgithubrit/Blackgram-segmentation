import cv2 as cv
import numpy as np

image_path = "BLACKGRAM-lower.png"
# Read the image using OpenCV
image = cv.imread(image_path)

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blur = cv.GaussianBlur(gray_image, (5, 5), 0)

# Apply binary threshold
_, binary_image = cv.threshold(blur, 70, 255, cv.THRESH_BINARY)

# Apply adaptive threshold
adaptive_image = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Apply Gaussian adaptive threshold
_, gaussian_image = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
gaussian_image = cv.GaussianBlur(gaussian_image, (5, 5), 0)

# Display the original, blurred, and thresholded images
cv.imshow('Original Image', image)
cv.imshow('Binary Thresholding', binary_image)
cv.imshow('Adaptive Thresholding', adaptive_image)
cv.imshow('Gaussian Thresholding', gaussian_image)

cv.waitKey(0)
cv.destroyAllWindows()


