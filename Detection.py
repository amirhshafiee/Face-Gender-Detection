# Develop by Amirhosien Shafiee
# Telegram: @amirhshafiee
# Email: amirshafiee266@yahoo.com


import glob
import cv2
import numpy as np
from joblib import load
from mtcnn import MTCNN

sgd = load("gender_classifier.z")

def face_detector(img):

    detector = MTCNN()
    img_ls = []
    points = []
    rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ls = detector.detect_faces(rbg_img)
    for faces in ls:
        x, y, w, h = faces["box"]
        points.append([x, y, w, h])
        img_ls.append(img[y:y+h, x:x+w])
    return img_ls, points


video = cv2.VideoCapture(0)

while True: 
    ret, frame = video.read()
    if frame is None:
        continue
    img_ls, points = face_detector(frame) 

    if img_ls == []:
        cv2.putText(frame, "not humen", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)
        img = cv2.resize(frame, (500, 500))
        cv2.imshow("photo", img)
    else:
        for i, face in enumerate(img_ls):
            face = cv2.resize(face, (32, 32))
            face = face.flatten()
            face = face/255
            
            out = sgd.predict(np.array([face]))[0]

            if out == "male":
                cv2.rectangle(frame, (points[i][0], points[i][1]), (points[i][0]+points[i][2], points[i][1]+points[i][3]), (0, 255, 0), 2)
                cv2.putText(frame, "male", (points[i][0], points[i][1]-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            elif out == "female":
                cv2.rectangle(frame, (points[i][0], points[i][1]), (points[i][0]+points[i][2], points[i][1]+points[i][3]), (0, 0, 255), 2)
                cv2.putText(frame, "female", (points[i][0], points[i][1]-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
    
        img = cv2.resize(frame, (500, 500))
        cv2.imshow("photo", img)
        if cv2.waitKey(1)  == ord("q"):
            break         

cv2.destroyAllWindows()

