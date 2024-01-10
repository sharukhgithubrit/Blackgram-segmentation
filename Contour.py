import cv2
import numpy as np

# Load the Blackgram image
image = cv2.imread("BLACKGRAM-lower.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and help contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection with adjusted parameters
edges = cv2.Canny(blurred, 20, 80)  # Adjust these values based on your image characteristics

# Find contours in the edged image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a copy of the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the contour image
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
