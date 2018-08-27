# -*- coding: utf-8 -*-
"""
PW3: Color layout descriptor
"""

#%%
import numpy as np
import cv2
import colorLayout
from matplotlib import pyplot as plt

#%% Load the image in colors
img = cv2.imread('peppers.bmp') 
cv2.imshow('Peppers', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
n_blocks_per_row = 128
n_blocks_per_col = 128
n_blocks = n_blocks_per_row*n_blocks_per_col;

tmp1 = np.zeros([n_blocks_per_row,n_blocks_per_col,3],dtype='uint8')

R,C, ch = np.shape(img)
indxR = np.arange(0,R,R/n_blocks_per_row) 
indxC = np.arange(0,C,C/n_blocks_per_col) 

for ir,r in enumerate(indxR):
    for ic,c in enumerate(indxC):
        block = img[r:r+R/n_blocks_per_row, c:c+C/n_blocks_per_col,:]
#        cv2.imshow('block',block)
#        cv2.waitKey(200)
#        cv2.destroyAllWindows()
        mean, std = cv2.meanStdDev(block)
        tmp1[ir,ic,:]= mean[:,0]

#%%    
cv2.namedWindow('blocks',cv2.WINDOW_NORMAL )    
cv2.imshow('blocks', tmp1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% 
ycbcr = cv2.cvtColor(tmp1,cv2.COLOR_BGR2YCR_CB)
y, cr, cb = cv2.split(ycbcr)

dy = cv2.dct(np.float32(y))

cv2.namedWindow('Y',cv2.WINDOW_NORMAL )
cv2.imshow('Y', y)
   
#%%
logdy= np.log10(np.abs(dy))
mmdy = np.min(logdy)
MMdy = np.max(logdy)
tmp =  cv2.applyColorMap(np.uint8(255* (logdy-mmdy)/(MMdy-mmdy)),cv2.COLORMAP_JET)
#%%
cv2.namedWindow('DCT-Y',cv2.WINDOW_NORMAL )  
cv2.imshow('DCT-Y',tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Compute descriptor

descr = colorLayout.clDescr(img)

print(descr)