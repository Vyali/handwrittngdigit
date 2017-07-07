# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from pylab import *


img=cv2.imread('digitimg.jpg',0)
size=50, 50
res=cv2.resize(img,size)

#ret,thresh = cv2.threshold(img,200,1,cv2.THRESH_BINARY)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)

im,thresh= cv2.threshold(img,200,1,cv2.THRESH_BINARY_INV)

'''
CannyImage = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(CannyImage,cmap= 'gray')
plt.title('Edge Image'),plt.xticks([]),plt.yticks([])
plt.show()
cv2.imshow('canny image',CannyImage)
ret,thresh = cv2.threshold(CannyImage,127,1,cv2.THRESH_BINARY)
# c++ code
#vector<vector<point>> contours
#vector<vect4i> heirarchy
'''
im,contours, heirarchy = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE);


'''
for i in range(0,len(contours)):
    if (heirarchy[0][i][3] >= 0 ):
        c=contours[i]
        cv2.drawContours(img, [c],0,(0, 255, 0), 1)
        #plt.title('canny image'),plt.imshow(CannyImage) 
        cv2.imshow('casd',img)

cv2.imshow('contrours image ', img)
'''


size=(28,28)

#for cnt in contours:
cnt=contours[5]    
x,y,w,h =cv2.boundingRect(cnt)
print('x')
print(x)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
cv2.imshow('recrtangel image ',img)
#cv2.waitKey(2000)
c=contours[5]
x,y,w,h =cv2.boundingRect(c)
crop= img[y:y+h,x:x+w]
res = cv2.resize(crop,size)
cv2.imshow('cropped image',crop)
cv2.imshow('resized image',res)

#gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(res,200,1,cv2.THRESH_BINARY)

print("len of thresh",len(thresh))
print("thresh value",thresh)



fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(thresh, (28,28) ), cmap=cm.gray )

def drawBiggerRect():
    for cnt in contours:
        x,y,w,h =cv2.boundingRect(cnt)
        l1=line([x,y],[x+w,y+h])
        l2=line([x+w,y],[x,y+h])
        x1,y1=intersection(l1,l2)
        for c in contours:
            xt,yt,wt,ht =cv2.boundingRect(cnt)
            l3=line([x,y],[xt+wt,yt+ht])
            l4=line([xt+wt,yt],[xt,yt+ht])
            x2,y2=intersection(l3,l4)
            
        

        
        
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False        
#plt.title('dfsscanny image'),plt.imshow(CannyImage)
#print (heirarchy[0][3][1])
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
path = 'handwrittingsample.jpg'
img = Image.open(path)
img = img.convert('RGBA')
pix = img.load()
for y in range(img.size[1]):
    for x in range(img.size[0]):
        if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
            pix[x, y] = (0, 0, 0, 255)
        else:
            pix[x, y] = (255, 255, 255, 255)
img.save('temp.jpg')
text = pytesseract.image_to_string(Image.open('temp.jpg'))
# os.remove('temp.jpg')
print(text)   
'''     
