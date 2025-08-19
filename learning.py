import cv2
import glob
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN
from joblib import dump



def face_detector(img):
    detector = MTCNN()
    img_ls = []
    rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ls = detector.detect_faces(rbg_img)
    for faces in ls:
        x, y, w, h = faces["box"]
        img_ls.append(img[y:y+h, x:x+w])
    return img_ls

data = []
lable = []

for item in glob.glob("gender\\*\\*"):
    img = cv2.imread(item)
    img_ls = face_detector(img)
    if img_ls == []:
        continue
    else:
        for face in img_ls:
            face = cv2.resize(face, (32, 32))
            face = face.flatten()
            face = face/255
            data.append(face)
            lable.append(item.split("\\")[1])



data = np.array(data)

x_train, x_test, y_train, y_test = train_test_split(data, lable, test_size=0.2)

sgd = SGDClassifier()

sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)

acc = accuracy_score(y_test, y_pred) * 100
print("Acc: %.2f"%acc)

dump(sgd, "gender_classifier.z")