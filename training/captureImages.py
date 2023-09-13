import cv2
import os
import numpy as np
from configuration import *




def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(3,3),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

def captureImages(coinName, cleanFolder=False):
    video = cv2.VideoCapture(f"http://{CAMERA_IP}:{CAMERA_PORT}/video")

    folder = f'imagesToTraining/{coinName}'
    if not os.path.exists(folder):
        # Se a pasta não existir, crie-a
        os.mkdir(folder)

    if cleanFolder:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                os.remove(path)

    index = 0
    while True:
        _,img = video.read()
        img = cv2.resize(img,(640,480))
        imgPre = preProcess(img)
        countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


        for cnt in countors:
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                recorte = img[y:y +h,x:x+ w]
                recorte = cv2.resize(recorte, (224, 224))
                cv2.imshow('IMG', recorte)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(f'{folder}/imagem{index}.jpg', recorte)
                    index += 1
                    print(f'Saving, Index: {index}')
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

captureImages('50_cents', True)