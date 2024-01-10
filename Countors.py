import cv2
import numpy as np

# Load the image
image = cv2.imread('WHEAT-lower.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours: {len(contours)}")

# Draw the contours on a copy of the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Denoise the image using a median filter
denoised_image = cv2.medianBlur(image, 5)

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Display the denoised image
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)

# Display the contour image
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

