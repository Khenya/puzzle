import numpy as np 
import cv2 

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Define the new color in formate RGB
new_color = (0, 0, 255)  # red

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Loop through all the pixels in the frame and change their color
    for y in range(height):
        for x in range(width):
            pixel = frame[y, x].astype('float')
            if (pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20) and (max(pixel[0:2]) - min(pixel[0:2]) > 15) and (
                    abs(pixel[2] - pixel[1]) > 15) and (pixel[2] > pixel[1] and pixel[2] > pixel[0]):
                frame[y, x] = new_color
            else:
                frame[y, x] = (0, 0, 0)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()