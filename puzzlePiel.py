import numpy as np 
import cv2 

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Define the new color in formate RGB
new_color = (255, 255, 255)  # red

def es_piel(rgb):
    return rgb[2]>95 and rgb[1]>40 and rgb[0]>20 and (max(rgb)-min(rgb)>15) and \
        abs(rgb[2]-rgb[1])>15 and rgb[2]>rgb[1] and rgb[2]>rgb[0]

# Cambiar para ajustar el tamaño de la pantalla
si = 600

# Cambiar para ajustar la precisión (1px,3px)
pr = 2

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(si+200,si), fx = 0.1, fy = 0.1)

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Loop through all the pixels in the frame and change their color
    for y in range(pr-1,height,pr):
        for x in range(pr-1,width,pr):
            pixel = frame[y, x].astype('float')
            if es_piel(pixel):
                frame[y-pr:y, x-pr:x] = new_color
            else:
                frame[y-pr:y, x-pr:x] = (0, 0, 0)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()