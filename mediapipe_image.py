import cv2
import numpy as np
import mediapipe as mp
import random
import os

if __name__ == '__main__':
        
    # Load Puzzle image, split image into pieces and randomly shuffle the frames
    imgx = cv2.imread(r'mano5.jpg')
    # imgx = cv2.resize(imgx,(si,si), fx = 0.1, fy = 0.1)
    # stack,x,shl=[],[],[]
    # for i in range(5):
    #         for j in range(5):
    #             shl.append(imgx[bl[i]:bl[i+1],bl[j]:bl[j+1]])
    # random.shuffle(shl)

    # Capture live video
    # cap = cv2.VideoCapture(0)
    
    # initialize the Hands class and store it in a variable
    mp_hands = mp.solutions.hands
    
    # Set hands function which will hold the landmarks points
    hands = mp_hands.Hands(static_image_mode=True)
    
    # Drawing function of hand landmarks on the image
    mp_drawing = mp.solutions.drawing_utils

    # Capture and process image frame from video
    # success, image = cap.read()
    # image=cv2.flip(image,1)
    # image = cv2.resize(image,(si+200,si), fx = 0.1, fy = 0.1)
    imageRGB = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    print(results.multi_hand_landmarks)
            
    # Locate spatial location of finger and mark image frame piece and swap pieces
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
            mp_drawing.draw_landmarks(imgx, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("output", imgx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cap.release()
