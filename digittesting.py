# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:07:10 2017
@author: Ayushc
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import handdigit
import pickle
import cv2





############image segmentation code#########

def get_contour_precedence(contour, cols):
    tolerance_factor = 60
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

        
mynet = handdigit.network([784,12,10])
mynet.load("MNIST-CrossEntropy-Network")
img=cv2.imread('digitimg.jpg',0)

im,thresh= cv2.threshold(img,200,1,cv2.THRESH_BINARY_INV)
##finding contours and sorting them
im,contours, heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
contours.sort(key=lambda x:get_contour_precedence(x, thresh.shape[1]))

#print(contours)


##background
bgd=cv2.imread('bgd.jpg',0)

bgd=cv2.resize(bgd,(56,56))
im,bgd= cv2.threshold(bgd,200,1,cv2.THRESH_BINARY)
#blank_image = np.ones((56,56), np.uint8)
#fig, ax = plt.subplots(1,2,figsize=(8,4))
#ax[0].matshow( np.reshape(blank_image, (56,56) ), cmap=cm.gray )


size=(28,28)
#iterating through each and every  contours
imgnr=0
for cnt in contours: 
    
    # bounding rectangle
    x,y,w,h =cv2.boundingRect(cnt)
    #crop and resize 
    crop= img[y:y+h,x:x+w]
    
    res = cv2.resize(crop,(35,35))
    ret,thresh=cv2.threshold(res,200,1,cv2.THRESH_BINARY_INV)
    bgd[12:47,12:47]=thresh
    #wrap = cv2.copyMakeBorder(res,10,10,10,10,cv2.BORDER_WRAP)
    t = cv2.resize(bgd,size)
    #print('thresh shape',len(thresh))
    b=t.ravel()
    prediction = mynet.feedforward(b)
    #print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, handdigit.testSet[1][imgnr], np.argmax(prediction)))
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].matshow( np.reshape(b, (28,28) ), cmap=cm.gray )
    ax[1].plot( prediction, lw=3 )
    ax[1].set_aspect(9)
    print("ans is ",np.argmax(prediction))
    imgnr=imgnr+1
###############################










### creating test result
'''
with open ('binary.txt', 'rb') as fp:
    inputList = pickle.load(fp)    
'''

         
#mynet = handdigit.network([784,100,10])
#mynet.load("MNIST-CrossEntropy-Network")

# Choose a random entry from the test-data.
#imgnr = np.random.randint(0,1000)
# Feed it trough the network to get our prediction
#print(len(handdigit.testSet[0][imgnr]))


#print('test test')


#print (handdigit.testSet[0][imgnr])


prediction = mynet.feedforward( handdigit.testSet[0][imgnr] )
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, handdigit.testSet[1][imgnr], np.argmax(prediction)))

#prediction = mynet.feedforward( handdigit.testSet[0][imgnr] )
#print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, handdigit.testSet[1][imgnr], np.argmax(prediction)))


# Show the image together with the output neurons
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(handdigit.testSet[0][imgnr], (28,28) ), cmap=cm.gray )
ax[1].plot( prediction, lw=3 )
ax[1].set_aspect(9)
