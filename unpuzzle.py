import cv2
import numpy as np
import mediapipe as mp
import random

npiezas = 3
patito = npiezas * npiezas

# ganaste???
def win():
    global stack,shl
    # shl[stack[0]],shl[stack[1]]

# Swap two frames in the image
def swap():
    global stack,shl
    if len(stack)==2:
        shl[stack[0]],shl[stack[1]]=shl[stack[1]],shl[stack[0]]
        stack=[]
        win()

# Identify spatial location and swap frames
def spacial_location_check(cx,cy):
    time_to_select = 100
    global x
    chk=-1
    for i in range(npiezas):
        for j in range(npiezas):
            chk+=1
            if cx in range(bxs[j,i,0],bxl[j,i,0]) and cy in range(bxs[j,i,1],bxl[j,i,1]):
                x.append([i,j])
                if x[0]!=x[-1]:
                    x=[]
                elif len(x)==time_to_select:
                    stack.append(chk)
                    if len(stack)>1:
                        swap()
                    x=[]

if __name__ == '__main__':
    bxs = np.full((npiezas, npiezas, 2), 0)
    bxl = np.full((npiezas, npiezas, 2), 0)
    si = 600
    bl = [0, si // npiezas, (si // npiezas) * 2, (si // npiezas) * 3, (si // npiezas) * 4, si]  # Divide en 5 partes iguales
    for i in range(npiezas):
        for j in range(npiezas):
            bxs[i, j] = (bl[i], bl[j])
            bxl[i, j] = (bl[i + 1], bl[j + 1])

        
    # Load Puzzle image, split image into pieces and randomly shuffle the frames
    imgx = cv2.imread(r'img.jpg')
    fin = cv2.VideoCapture(r'boom.gif')
    
    imgx = cv2.resize(imgx,(si,si), fx = 0.1, fy = 0.1)
    stack,x,shl=[],[],[]
    for i in range(npiezas):
        for j in range(npiezas):
            shl.append(imgx[bl[i]:bl[i+1],bl[j]:bl[j+1]])
    random.shuffle(shl)

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
        success, f = fin.read()

        success, image = cap.read()
        image=cv2.flip(image,1)
        image = cv2.resize(image,(si+200,si), fx = 0.1, fy = 0.1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        ck=0

        # Draw pieces over image
        for i in range(npiezas):
            for j in range(npiezas):
                if ck in stack:
                    image[bl[i]:bl[i+1],bl[j]:bl[j+1]]=cv2.blur(shl[ck],ksize=(15,15))
                else:
                    image[bl[i]:bl[i+1],bl[j]:bl[j+1]]=shl[ck]
                ck+=1
                
        # Locate spatial location of finger and mark image frame piece and swap pieces
        if results.multi_hand_landmarks:
            cxi = 100000
            cxp = 100000 
            cyi = 0
            cyp = 0 
            for handLms in results.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark):
                    if id == 8 :
                        h, w, c = image.shape
                        cxi, cyi = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (cxi, cyi), 25, (255, 0, 0), cv2.FILLED)
                    if id == 4:
                        h, w, c = image.shape
                        cxp, cyp = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (cxp, cyp), 25, (255, 0, 0), cv2.FILLED)
                    
                    distance = np.sqrt((cxi - cxp)**2 + (cyi - cyp)**2)
                    if distance < 50 and distance > 0:
                        cv2.circle(image, (cxi, cyi), 25, (0, 0, 255), cv2.FILLED)
                        cv2.circle(image, (cxp, cyp), 25, (0, 0, 255), cv2.FILLED)
                        spacial_location_check(cxi,cyi)

                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

        if type(f) != type(None):
            cv2.imshow("output", f)
        cv2.imshow("output", image)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
    cap.release()
