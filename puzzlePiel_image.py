import numpy as np 
import cv2 

# Capture video from the camera
# cap = cv2.VideoCapture(0)

# Define the new color in formate RGB
new_color = (255, 255, 255)  # red

def es_piel(rgb):
    return rgb[2]>95 and rgb[1]>40 and rgb[0]>20 and (max(rgb)-min(rgb)>15) and \
        abs(rgb[2]-rgb[1])>15 and rgb[2]>rgb[1] and rgb[2]>rgb[0]

# Cambiar para ajustar el tamaño de la pantalla
si = 600

# Cambiar para ajustar la precisión (1px,3px)
pr = 1

imgx = cv2.imread(r'mano5.jpg')

# Get the height and width of the frame
height, width, _ = imgx.shape

# Loop through all the pixels in the frame and change their color
for y in range(pr-1,height,pr):
    for x in range(pr-1,width,pr):
        pixel = imgx[y, x].astype('float')
        if es_piel(pixel):
            imgx[y-pr:y, x-pr:x] = new_color
        else:
            imgx[y-pr:y, x-pr:x] = (0, 0, 0)

# Display the resulting frame
cv2.imshow('Frame', imgx)

# Exit the loop when the 'q' key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()