# -*- coding: utf-8 -*-
"""
PW1: basic descriptors
"""

#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt


#%% Load an color image in grayscale
img1 = cv2.imread('coffee.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('house.jpg',cv2.IMREAD_GRAYSCALE)

#%% Show the images
cv2.imshow('Image 1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Image 2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Compute basic descriptors
mean1, std1 = cv2.meanStdDev(img1)  # mean, standard deviation
mean2, std2 = cv2.meanStdDev(img2)

plt.plot(mean1, std1, 'o',mean2, std2, 'x'  )
plt.axis([0, 255, 0, 255])
plt.xlabel('Mean')
plt.ylabel('Std')

#%%
distance = np.sqrt( (mean1-mean2)**2 + (std1-std2)**2 )
print "Distance = %5.2f"% distance

#%% Compute similar images in a set
image_files = ( 'baboon.jpg',  'bottle.jpg',  'bridge.jpg' , 'cameraman.jpg', 'clown.jpg',  'coffee.jpg', 'couple.jpg', 'crowd.jpg', 'einst.jpg', 'house.jpg', 'lake.jpg', 'lax.jpg',   'lena.jpg',  'man.jpg',  'peppers.jpg',  'plane.jpg',  'spring.jpg',   'trees.jpg',   'truck.jpg', 'egg.jpg',   'bird.jpg',   'woman1.jpg',  'woman2.jpg', 'zelda.jpg' )
descr = np.zeros([len(image_files), 2])


for ind1,name1 in enumerate(image_files):
    img1 = cv2.imread(name1,cv2.IMREAD_GRAYSCALE)
    mean1, std1 = cv2.meanStdDev(img1)
    descr[ind1,0] = mean1
    descr[ind1,1] = std1     
#%%   

plt.plot(descr[:,0],descr[:,1],'x')
plt.axis([0, 255, 0, 100])
plt.xlabel('Mean')
plt.ylabel('Std')


#%%      
distances = np.zeros([len(image_files),len(image_files)])   
di =  np.diag_indices(len(image_files))    
DD = np.matmul(descr,np.transpose(descr))
for i in range(len(image_files)):
    for j in np.arange(i+1,len(image_files)):        
        distances[i,j] = DD[i,i]+DD[j,j] - DD[i,j]-DD[j,i]
        distances[j,i] = distances[i,j]

distances[di] = distances.max()
#%%

logdi =  np.log10(distances)
nlogdi = (logdi-logdi.min())/(logdi.max()-logdi.min())
cv2.namedWindow('Distances', cv2.WINDOW_NORMAL)
cv2.imshow('Distances', nlogdi)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
i1, i2 = np.unravel_index(distances.argmin(), distances.shape)
img1 = cv2.imread(image_files[i1],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image_files[i2],cv2.IMREAD_GRAYSCALE)              
cv2.imshow('Image 1',img1)
cv2.imshow('Image 2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()   


#%%   

plt.plot(descr[:,0],descr[:,1],'x', descr[i1,0],descr[i1,1],'o', descr[i2,0],descr[i2,1],'o')
plt.axis([0, 255, 0, 100])
plt.xlabel('Mean')
plt.ylabel('Std')

#%% Local mean and std

n_blocks_per_row = 2;
n_blocks_per_col = 2;
n_blocks = n_blocks_per_row*n_blocks_per_col;

descr = np.zeros([len(image_files), 2*n_blocks])


for ind1,name1 in enumerate(image_files):
    img1 = cv2.imread(name1,cv2.IMREAD_GRAYSCALE)
    
    R,C = np.shape(img1)
    indxR = np.arange(0,R,R/n_blocks_per_row) 
    indxC = np.arange(0,C,C/n_blocks_per_col) 
    print  "%15s" % name1
    
    for ir,r in enumerate(indxR):
        for ic,c in enumerate(indxC):
            block = img1[r:r+R/n_blocks_per_row, c:c+C/n_blocks_per_col]
#            cv2.imshow('block',block)
#            cv2.waitKey(100)
#            cv2.destroyAllWindows()
            mean, std = cv2.meanStdDev(block)
            descr[ind1,2*(ic+ir*n_blocks_per_col)]=mean 
            descr[ind1,2*(ic+ir*n_blocks_per_col)+1]= std 
            print "Image %d" % ind1 , "Block %d " % (ic+ir*n_blocks_per_col) , 
            print "%5.2f" % mean, "%5.2f" % std
    
    
    #%%      
distances = np.zeros([len(image_files),len(image_files)])   
di =  np.diag_indices(len(image_files))    
DD = np.matmul(descr,np.transpose(descr))
for i in range(len(image_files)):
    for j in np.arange(i+1,len(image_files)):        
        distances[i,j] = DD[i,i]+DD[j,j] - DD[i,j]-DD[j,i]
        distances[j,i] = distances[i,j]

distances[di] = distances.max()
#%%

logdi =  np.log10(distances)
nlogdi = (logdi-logdi.min())/(logdi.max()-logdi.min())
cv2.namedWindow('Distances', cv2.WINDOW_NORMAL)
cv2.imshow('Distances', nlogdi)
cv2.waitKey(0)
cv2.destroyAllWindows()


 #%%
i1, i2 = np.unravel_index(distances.argmin(), distances.shape)
img1 = cv2.imread(image_files[i1],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image_files[i2],cv2.IMREAD_GRAYSCALE)              
cv2.imshow('Image 1',img1)
cv2.imshow('Image 2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()   

