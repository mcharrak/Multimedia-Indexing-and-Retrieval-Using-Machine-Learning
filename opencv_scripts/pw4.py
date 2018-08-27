  # -*- coding: utf-8 -*-
"""
Harris corner detector

"""


import cv2
import numpy as np

filename = 'bottle.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

blockSize = 2
sobelParam = 3
alpha = 0.01
dst = cv2.cornerHarris(gray,blockSize,sobelParam,alpha)

#result is dilated for marking the corners 
dst = cv2.dilate(dst,None)

# Threshold, it may vary depending on the image.
threshold = 0.015*dst.max()
color = [0,0,255] # it is red

img[dst>threshold]=color

cv2.imshow('dst',img)
cv2.waitKey(0)  
cv2.destroyAllWindows() 

#%% Test feature invariance to rotation

rows,cols = gray.shape 
M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
rotated = np.uint8(cv2.warpAffine(gray,M,(cols,rows)))

cv2.imshow('rotated',rotated)
cv2.waitKey(0)  
cv2.destroyAllWindows() 

cv2.imwrite('rotated.jpg',rotated)

filename = 'rotated.jpg'
img2 = cv2.imread(filename)
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

blockSize = 2
sobelParam = 3
alpha = 0.01
dst = cv2.cornerHarris(gray,blockSize,sobelParam,alpha)

#result is dilated for marking the corners 
dst = cv2.dilate(dst,None)

# Threshold, it may vary depending on the image.
# Dont change the threshold 
color = [0,0,255] # it is red

img2[dst>threshold]=color

cv2.imshow('img1',img)
cv2.imshow('img2',img2)
cv2.waitKey(0)  
cv2.destroyAllWindows() 


#%% Test invariance to zoom
filename = 'bottle.jpg'
img3 = cv2.imread(filename)
 

zoomed = cv2.resize(img3,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('zoom',zoomed)
cv2.waitKey(0)  
cv2.destroyAllWindows() 

gray = cv2.cvtColor(zoomed,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

blockSize = 2
sobelParam = 3
alpha = 0.01
dst = cv2.cornerHarris(gray,blockSize,sobelParam,alpha)

#result is dilated for marking the corners 
dst = cv2.dilate(dst,None)

# Threshold, it may vary depending on the image.
threshold = 0.015*dst.max()
color = [0,0,255] # it is red

zoomed[dst>threshold]=color

cv2.imshow('img1',img)
cv2.imshow('Zoomed',zoomed)
cv2.waitKey(0)  
cv2.destroyAllWindows() 
