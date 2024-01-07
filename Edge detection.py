import cv2 as cv

# Read the image
image = cv.imread('BLACKGRAM-lower.png')

if image is not None:
    # Convert the image to RGB for displaying with matplotlib
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Apply edge detection using Canny
    edges = cv.Canny(image, 75, 150)
    
    # Display the original and edge images using cv.imshow
    cv.imshow('Original Image', rgb_image)
    cv.imshow('Edges', edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
