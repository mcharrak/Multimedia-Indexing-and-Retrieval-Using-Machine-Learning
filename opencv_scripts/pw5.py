# -*- coding: utf-8 -*-
"""
SIFT
"""

import cv2
import numpy as np

img = cv2.imread('house.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.FeatureDetector_create("ORB") #%% SFIT crashes!
features = cv2.SIFT()
kp = features.detect(gray,None)
   
img=cv2.drawKeypoints(gray,kp,img)

#cv2.imwrite('sift_keypoints.jpg',img)

cv2.imshow('features', img)
cv2.waitKey(0)


#%% Test feature invariance to rotation

rows,cols = gray.shape 
M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
rotated = np.uint8(cv2.warpAffine(gray,M,(cols,rows)))

cv2.imshow('rotated',rotated)
cv2.waitKey(0)  


cv2.imwrite('rotated.jpg',rotated)

filename = 'rotated.jpg'
img2 = cv2.imread(filename)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

features2 = cv2.FeatureDetector_create("ORB") #%% SFIT crashes!
kp2 = features.detect(gray2,None)
   
img2=cv2.drawKeypoints(gray2,kp2,img2)

#cv2.imwrite('sift_keypoints.jpg',img)

cv2.imshow('features rotation', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

 