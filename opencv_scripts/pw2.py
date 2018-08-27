# -*- coding: utf-8 -*-
"""
PW2: colorspaces
"""

#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
#%% Load the image in colors
img = cv2.imread('coffee.bmp') # It is the default behavior

#%% Show the color image
cv2.imshow('Color image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Show the R G B components
cv2.imshow('Blue channel',img[:,:,0])

cv2.imshow('Green channel',img[:,:,1])

cv2.imshow('Red channel',img[:,:,2])

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Color histograms

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

#%% Changing color space
img = cv2.imread('peppers.bmp') # It is the default behavior

# change to the HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('Color image',img)
cv2.imshow('Hue channel',hsv[:,:,0])

cv2.imshow('Saturation channel',hsv[:,:,1])

cv2.imshow('Value channel',hsv[:,:,2])

cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
color = ('c','m','k')
for i,col in enumerate(color):
    histr = cv2.calcHist([hsv],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.legend(['h', 's', 'v '])
plt.show()


#%% show the green part of the image


# define range of green color in HSV
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
GR = hsv_green[0,0,0]

eps = 20
lower_green = np.array([GR-eps,50,50])
upper_green = np.array([GR+eps,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow('image',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

