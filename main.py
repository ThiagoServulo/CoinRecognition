import cv2
import numpy as np
from keras.models import load_model
from src import PreProcessing, Ellipse


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
#camera = cv2.VideoCapture(0)

#video = cv2.VideoCapture("http://192.168.0.2:4748/video")

index22 = 0

while True:
    if index22 == 0:
        index22 = 1
        # Grab the webcamera's image.
        #ret, image = video.read()
        image = cv2.imread('training/imagesToTraining/teste/imagem711.jpg')
        #amarela(image)
        img_back = image.copy()
        img_result =  image.copy()

        imgPre = PreProcessing.ImagePreProcessing(image)
        count = Ellipse.RecognizeEllipses(image.copy(), imgPre)

        cv2.imshow('recorte', imgPre)
        countors, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        print(f'Tamanho: {len(count)}')
        qtd = 0
        for cnt in count:
            print('aaaaa')
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                img2 = img_back[y:y + h, x:x + w]
                img2 = cv2.GaussianBlur(img2, (3, 3), 3)
                cv2.imshow('recorte', img2)

                # Resize the raw image into (224-height,224-width) pixels
                img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)

                # Show the image in a window
                #cv2.imshow("Webcam Image", image)

                # Make the image a numpy array and reshape it to the models input shape.
                img2 = np.asarray(img2, dtype=np.float32).reshape(1, 224, 224, 3)

                # Normalize the image array
                img2 = (img2 / 127.5) - 1

                # Predicts the model
                prediction = model.predict(img2)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # Print prediction and confidence score
                classe = class_name[2:]

                conf = str(np.round(confidence_score * 100))[:-2]

                if int(conf) > 50:
                    print("Class:", classe, end="")
                    print("Confidence Score:", conf, "%")

                    cv2.putText(img_result, f'{classe}: {conf}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Resultado", img_result)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

#camera.release()
cv2.destroyAllWindows()



