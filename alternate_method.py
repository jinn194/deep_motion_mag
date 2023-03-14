import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video file
cap = cv2.VideoCapture('baby.mp4')

# Get the frame rate and size of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction to detect movement
    fg_mask = bg_subtractor.apply(frame)
    
    # Apply a threshold to the foreground mask to extract the moving pixels
    thresh = 25
    fg_mask[fg_mask < thresh] = 0
    fg_mask[fg_mask >= thresh] = 255
    
    # Apply a dilation operation to exaggerate the moving pixels
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
    
    # Apply a blur to smooth out the edges of the moving pixels
    blur = cv2.GaussianBlur(fg_mask, (21, 21), 0)
    
    # Blend the original frame with the exaggerated foreground mask
    alpha = 0.5
    frame = cv2.addWeighted(frame, alpha, cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), 1 - alpha, 0)
    
    # Write the output frame to the video file
    out.write(frame)
    
    # Display the output frame in a window
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()