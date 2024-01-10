import cv2
import numpy as np

# Load the image
image = cv2.imread("BLACKGRAM-lower.png")
original_image = image.copy()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection with binary result
edges = cv2.Canny(blurred, 30, 90)

# Apply a binary threshold to the edge image
threshold_value = 60
_, binary_edges = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours in the binary edge image
contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define parameters for sector area thresholding
R = 20  # Radius of the circle
A_threshold = 20  # Acute angle threshold

# Iterate through contours
for contour in contours:
    # Approximate the contour to reduce the number of points
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour is convex
    if cv2.isContourConvex(approx):
        continue

    # Calculate the centroid of the contour
    M = cv2.moments(approx)
    if M["m00"] == 0:
        continue
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Check each point in the contour
    for point in approx:
        x, y = point[0]
        # Calculate the angle from the centroid to the current point
        angle = np.degrees(np.arctan2(y - cY, x - cX))

        # Calculate the distance from the centroid to the current point
        distance = np.sqrt((x - cX)**2 + (y - cY)**2)

        # Check if the point is within the specified sector area
        if distance <= R and 0 <= angle <= A_threshold:
            # Draw a circle around the concave point on the original image
            cv2.circle(original_image, (x, y), 5, (0, 255, 0), -1)

# Display the original image with detected concave points overlaid on binary edges
cv2.imshow('Detected Concave Points on Binary Edges', original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


