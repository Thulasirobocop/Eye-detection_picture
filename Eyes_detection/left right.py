import cv2
import numpy as np
cascade=cv2.CascadeClassifier('C:\\Users\\sthul\\Downloads\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_lefteye_2splits.xml')
img=cv2.imread('D:\PROJECT\Eyes_detection\\i.jpg')

copy=img.copy()
gray=cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
eyes=cascade.detectMultiScale(gray,1.3,5)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(copy,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
    
cascade1=cv2.CascadeClassifier('C:\\Users\\sthul\\Downloads\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_righteye_2splits.xml')
copy1=img.copy()
gray=cv2.cvtColor(copy1,cv2.COLOR_BGR2GRAY)
eyes1=cascade1.detectMultiScale(gray,1.3,5)
for (ex,ey,ew,eh) in eyes1:
    cv2.rectangle(copy1,(ex,ey),(ex+ew,ey+eh),(255,0,0),3)

stack=np.vstack([img,copy,copy1])
stack=cv2.resize(stack,(500,500))
cv2.imshow('Output', stack)
cv2.waitKey(0)

