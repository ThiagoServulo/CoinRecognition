import cv2
import numpy as np
from keras.models import load_model

#tf 2.9.1
#keras 2.6.0

video = cv2.VideoCapture("http://192.168.0.154:4747/video")

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(3,3),5)
    imgPre = cv2.Canny(imgPre,100,200)
    kernel = np.ones((2,2),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre


model = load_model('Keras_model.h5',compile=False)
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
classes = ["10 cents", "50 cents"]

def DetectarMoeda(img):
    imgMoeda = cv2.resize(img,(224,224))
    imgMoeda = np.asarray(imgMoeda)
    imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
    data[0] = imgMoedaNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe,percent

"""
def DetectarMoeda(img):
    imgMoeda = cv2.resize(img,(224,224))
    imgMoeda = np.asarray(imgMoeda)
    imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
    data[0] = imgMoedaNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe,percent



while True:
    _,img = video.read()
    img = cv2.resize(img,(640,480))
    imgPre = preProcess(img)
    countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    qtd = 0
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            recorte = img[y:y +h,x:x+ w]
            classe, conf = DetectarMoeda(recorte)
            if conf >0.7:
                cv2.putText(img,str(classe),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                if classe == '1 real': qtd+=1
                if classe == '25 cent': qtd += 0.25
                if classe == '50 cent': qtd += 0.5

    cv2.rectangle(img,(430,30),(600,80),(0,0,255),-1)
    cv2.putText(img,f'R$ {qtd}',(440,67),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

    cv2.imshow('IMG',img)
    cv2.imshow('IMG PRE', imgPre)
    cv2.waitKey(1)

"""
quantidade = 0
total = 0
index = 0
while True:
    _,img = video.read()
    img = cv2.resize(img,(640,480))
    imgPre = preProcess(img)

    countors, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in countors:
        # Aproximar o contorno por um polígono
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Se o polígono aproximado tiver 5 ou 6 vértices (ou seja, é uma forma semelhante a um círculo)
        if len(approx) >= 5:
            # Desenhar o contorno em verde (ou realizar outra ação)
            contour_image = cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv2.imshow('cont', contour_image)
            print(len(approx))

    #qtd = 0
    #for cnt in countors:
    #    area = cv2.contourArea(cnt)
    #    if area > 2000:
    #        x, y, w, h = cv2.boundingRect(cnt)
    #        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #        recorte = img[y:y + h, x:x + w]
    #        classe, conf = DetectarMoeda(recorte)
    #        if conf > 0.9:
    #            cv2.putText(img,f'{classe} {conf}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    #            #if classe == '1 real': qtd+=1
    #            #if classe == '25 cent': qtd += 0.25
    #            #if classe == '50 cent': qtd += 0.5
#
    #        if quantidade < 100:
    #            quantidade += 1
    #            total += conf



    cv2.imshow('IMG PRE', imgPre)
    cv2.imshow('IMG',img)

    cv2.waitKey(1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        print(f'{quantidade} - {total/quantidade}')
        break


"""
            key = cv2.waitKey(1) & 0xFF
            # Se a tecla 's' for pressionada, faça algo
            recorte = cv2.resize(recorte, (224, 224))
            cv2.imshow('IMG', recorte)

            if key == ord('s'):
                # Faça algo aqui, por exemplo, exibir uma mensagem

                cv2.imwrite(f'imagesToTraining/imagem{index}.jpg', recorte)
                print('salvou')
                index += 1
"""
