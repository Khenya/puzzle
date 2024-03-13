import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Swap two frames in the image
def swap():
    global stack,shl
    if len(stack)==2:
        shl[stack[0]],shl[stack[1]]=shl[stack[1]],shl[stack[0]]
        stack=[]

# Identify spatial location and swap frames
def spacial_location_check(cx,cy):
    global x
    x = []  # Inicializar la lista x
    chk=-1
    for i in range(5):
        for j in range(5):
            chk+=1
            if cx in range(bxs[j,i,0],bxl[j,i,0]) and cy in range(bxs[j,i,1],bxl[j,i,1]):
                x.append([i,j])
                if x[0]!=x[-1]:
                    x=[]
                elif len(x)==25:
                    stack.append(chk)
                    if len(stack)>1:
                        swap()
                    x=[]

if __name__ == '__main__':
    bxs = np.full((5, 5, 2), 0)
    bxl = np.full((5, 5, 2), 0)
    si = 600
    bl = [0, si // 5, (si // 5) * 2, (si // 5) * 3, (si // 5) * 4, si]  # Divide en 5 partes iguales
    for i in range(5):
        for j in range(5):
            bxs[i, j] = (bl[i], bl[j])
            bxl[i, j] = (bl[i + 1], bl[j + 1])

    # Load Puzzle image, split image into pieces and randomly shuffle the frames
    imgx = cv2.imread(r'img.jpg')
    imgx = cv2.resize(imgx,(si,si), fx = 0.1, fy = 0.1)
    stack,x,shl=[],[],[]
    for i in range(5):
            for j in range(5):
                shl.append(imgx[bl[i]:bl[i+1],bl[j]:bl[j+1]])
    random.shuffle(shl)

    # Initialize the PyTorch model for hand keypoint detection
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Transform to apply to the input image
    transform = transforms.Compose([transforms.ToTensor()])

    # Capture live video
    cap = cv2.VideoCapture(0)

    while True:
        # Capture and process image frame from video
        success, image = cap.read()
        image=cv2.flip(image,1)
        image = cv2.resize(image,(si+200,si), fx = 0.1, fy = 0.1)
        
        # Convert image to torch tensor
        input_image = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            prediction = model(input_image)[0]

        # Extract hand keypoints from prediction
        boxes = prediction['boxes'].detach().cpu().numpy()
        keypoints = prediction['keypoints'].detach().cpu().numpy()

        # Draw rectangles around the detected hands and identify their spatial location
        for box in boxes:
            x, y, w, h = box.astype(int)
            cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
            cx, cy = (x + w) // 2, (y + h) // 2
            spacial_location_check(cx, cy)

        ck=0
        for i in range(5):
            for j in range(5):
                if ck in stack:
                    image[bl[i]:bl[i+1],bl[j]:bl[j+1]]=cv2.blur(shl[ck],ksize=(15,15))
                else:
                    image[bl[i]:bl[i+1],bl[j]:bl[j+1]]=shl[ck]
                ck+=1
        
        cv2.imshow("output", image)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    cv2.destroyAllWindows()
    cap.release()
