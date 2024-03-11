import cv2
import numpy as np
import mediapipe as mp

if __name__ == '__main__':
    bxs=np.full((3,3,2),0)
    bxl=np.full((3,3,2),0)
    si=600
    bl=[0,si//3,(si//3)*2,si]
    for i in range(3):
        for j in range(3):
            bxs[i,j]=(bl[i],bl[j])
            bxl[i,j]=(bl[i+1],bl[j+1])

    # Capture live video
    cap = cv2.VideoCapture(0)
    
    # initialize the Hands class and store it in a variable
    mp_hands = mp.solutions.hands
    
    # Set hands function which will hold the landmarks points
    hands = mp_hands.Hands(static_image_mode=True)
    
    # Drawing function of hand landmarks on the image
    mp_drawing = mp.solutions.drawing_utils

    while True:
        # Capture and process image frame from video
        success, image = cap.read()
        image=cv2.flip(image,1)
        image = cv2.resize(image,(si+200,si), fx = 0.1, fy = 0.1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        ck=0
                
        # Locate spatial location of finger and mark image frame piece and swap pieces
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: 
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("output", image)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    cv2.destroyAllWindows()
    cap.release()
