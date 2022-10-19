import cv2
import numpy as np
import os
import time 


path = 'final'
orb = cv2.ORB_create(nfeatures=100) #nfeatures=1000
########### import
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
#print('Total classees', len(myList))

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#######find descriptors of the images

def findDes(images):
    desList=[]
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

# we do not need to draw keypints, we have done detectino part above

#d ecripstor of cuurent frame from webcam
def findID(img, desList): 
    #des 2 is webcam, des is test images descriptors
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList=[] #list has no. of good images matching
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:  #*****
                    good.append([m])
            print(len(good))
            matchList.append(len(good))
        print("matchList appended")
    except:
        pass
    thresh = 5 #thresh = nearest match number
    if len(matchList):
        if max(matchList) > thresh: 
            finalVal = matchList.index(max(matchList))
    return finalVal       


############################# main functino :

desList = findDes(images)
print(len(desList))

img2 = cv2.imread(r'C:\Users\Lenovo\OneDrive\Desktop\test2\t2.png')
#cap = cv2.VideoCapture(0)
#While True:
    #success, img2 = cap.read()
    #imgOriginal = img2.copy()
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #id=findID(img2, desList)
     
    #if id!=-1:
    #   cv2.putText(img2, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    #send=classNames[id]
    #cv2.imshow('img2', img2)
    #cv2.waitKey(1)  

#image input
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
id = findID(img2, desList)
print(id)
if id!= -1:
    cv2.putText(img2, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow('img2', img2)
    
    send=classNames[id]
    
    print("...")
    print(send)
    print("...")
    cv2.waitKey(1)