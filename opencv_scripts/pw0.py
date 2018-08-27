# -*- coding: utf-8 -*-
"""
PW0: load and show an image
"""

#%%
import cv2

#%% Load an color image in grayscale
img = cv2.imread('coffee.bmp',cv2.IMREAD_GRAYSCALE)

#%% Show the grayscale image
cv2.imshow('Gray scale "coffee" image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Write the image
cv2.imwrite('housegray.png',img)

#%% Load the image in colors
img = cv2.imread('coffee.bmp') # It is the default behavior

#%% Show the color image
cv2.imshow('Color image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

