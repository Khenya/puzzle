import numpy as np 
import cv2 
import random

def find_contour_top(contour):
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    return topmost

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
    # Inicializar la captura de video en vivo
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.flip(frame,1)
        frame = cv2.resize(frame,(si+200,si), fx = 0.1, fy = 0.1)

        # Dibujar las fichas del puzzle
        ck=0
        for i in range(5):
            for j in range(5):
                if ck in stack:
                    frame[bl[i]:bl[i+1],bl[j]:bl[j+1]]=cv2.blur(shl[ck],ksize=(15,15))
                else:
                    frame[bl[i]:bl[i+1],bl[j]:bl[j+1]]=shl[ck]
                ck+=1
        
        # Convertir el frame de BGR a HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir el rango de color de la piel en HSV
        lower_skin = np.array([0, 20, 30], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Segmentar la imagen para aislar el color de la piel
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Encontrar contornos en la máscara de la piel
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si se detecta al menos un contorno
        if contours:
            # Tomar el contorno más grande (la mano)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Obtener el punto más alto del contorno de la mano
            top_point = find_contour_top(max_contour)
            
            # Si se encuentra el punto más alto de la mano
            if top_point:
                tx, ty = top_point
                # Buscar si la mano está sobre alguna ficha del puzzle
                for i in range(5):
                    for j in range(5):
                        x1, y1 = bxs[i, j]
                        x2, y2 = bxl[i, j]
                        if x1 < tx < x2 and y1 < ty < y2:
                            # Si la mano está sobre una ficha, mover la ficha al centro de la mano
                            stack = [i*5 + j]
                            break
        
        
        
        # Aplicar un poco de suavizado para eliminar el ruido
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Mostrar el frame original y el resultado de la segmentación
        cv2.imshow('Original', frame)
        cv2.imshow('Skin Detection', mask)
        
        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar los recursos y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
