import cv2
import os
dataset = 'dataset'
name = input('Enter your name for training Purpose:')
path = os.path.join(dataset,name)
print(path)
if not os.path.isdir(path):
    os.makedirs(path)

(width,height) = (130,100)
algo = 'haarcascade_frontalface_default.xml'
load_algo = cv2.CascadeClassifier(algo)
vid = cv2.VideoCapture(0)

count = 1
while count < 101:
    _,img = vid.read()
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = load_algo.detectMultiScale(grayimg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceonly = grayimg[y:y+h,x:x+w]
        resizeImg = cv2.resize(faceonly,(width,height))
        cv2.imwrite("{0}/{1}.jpg".format(path,count),resizeImg)
        count+=1
    cv2.imshow('Face Detection',img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 


