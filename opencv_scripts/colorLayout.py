# -*- coding: utf-8 -*-
"""
ColorLayout descriptor (MPEG7)
"""
import numpy as np
import cv2

def clDescr(img):
    #%%
    n_blocks_per_row = 8
    n_blocks_per_col = 8
    
    tmp1 = np.zeros([n_blocks_per_row,n_blocks_per_col,3],dtype='uint8')
    
    R,C, ch = np.shape(img)
    indxR = np.arange(0,R,R/n_blocks_per_row) 
    indxC = np.arange(0,C,C/n_blocks_per_col) 
    
    for ir,r in enumerate(indxR):
        for ic,c in enumerate(indxC):
            block = img[r:r+R/n_blocks_per_row, c:c+C/n_blocks_per_col,:]

            mean, std = cv2.meanStdDev(block)
            tmp1[ir,ic,:]= mean[:,0]
    
    ycbcr = cv2.cvtColor(tmp1,cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycbcr)
    dy  = cv2.dct(np.float32(y))
    dcr = cv2.dct(np.float32(cr))
    dcb = cv2.dct(np.float32(cb))
    
    cld = np.ndarray([18,1])
    cld[0]=dy[0,0]
    cld[1]=dy[0,1]
    cld[2]=dy[1,0]
    cld[3]=dy[2,0]
    cld[4]=dy[1,1]
    cld[5]=dy[0,2]
    cld[6]=dcb[0,0]
    cld[7]=dcb[0,1]
    cld[8]=dcb[1,0]
    cld[9]=dcb[2,0]
    cld[10]=dcb[1,1]
    cld[11]=dcb[0,2]
    cld[12]=dcr[0,0]
    cld[13]=dcr[0,1]
    cld[14]=dcr[1,0]
    cld[15]=dcr[2,0]
    cld[16]=dcr[1,1]
    cld[17]=dcr[0,2]
    return cld
    