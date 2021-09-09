import cv2
import numpy as np
import os
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'dataset'
(images,labels,name,id) = ([],[],{},0)
for (subdir,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        name[id] = subdir
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath+'/'+filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1

(images,labels) = [np.array(lst) for lst in [images,labels]]
(width,height) = (130,100)
#model = cv2.face.FisherFaceRecognizer_create()
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)
vid = cv2.VideoCapture(0)

count = 1
while True:
    _,img = vid.read()
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(grayimg,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceonly = grayimg[y:y+h,x:x+w]
        face_resize = cv2.resize(faceonly,(width,height))
        prediction = model.predict(face_resize)
        print(prediction)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<80:
            cv2.putText(img,'{0}'.format(name[prediction[0]]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            count = 0
        else:
            count+=1
            cv2.putText(img,'Unknown Person',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if(count>100):
                print("Unknown Person")
                count = 0
    cv2.imshow('Face Detection',img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 