import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    # Resize the frame
    rescaled_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    
    return rescaled_frame

# Read the original image
original_frame = cv.imread("BLACKGRAM-lower.png")

# Call the rescaleFrame function
rescaled_frame = rescaleFrame(original_frame, scale=0.5)

# Display the original and rescaled frames
cv.imshow('Original Frame', original_frame)
cv.imshow('Rescaled Frame', rescaled_frame)

# Wait for a key press and close the windows
cv.waitKey(0)
cv.destroyAllWindows()
