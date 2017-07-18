'''
for generating images of 28 x 28 to create the trainign set
'''

import cv2
import scipy
import os
from PIL import Image
size=(28,28)
'''
dirc = os.path.dirname("img2")
if not os.path.exists(dirc):
       os.makedirs(dirc)
 '''      
for i in range(11,63):
    '''dirpath="img2/Samb0"+str("{0:0=2d}".format(i))
    directory = os.path.dirname(dirpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
     '''   
    for j in range(1,56):
        path = "Img/Sample0"+str("{0:0=2d}".format(i))+"/img0"+str("{0:0=2d}".format(i))+"-0"+str("{0:0=2d}".format(j))+".png"
        #path1 = "Img1/Samb0"+str("{0:0=2d}".format(i))+"img0"+str("{0:0=2d}".format(i))+"-0"+str("{0:0=2d}".format(j))
        path1='image'+str(i)+"-"+str(j)+".png"
        img=cv2.imread(path)
        #im=Image.open(path)
        res = cv2.resize(img,size)
        cv2.imwrite(path1,res)
        #im.thumbnail(size,Image.ANTIALIAS)
        #im.save(path1,"png")
        cv2.imwrite(path1,res)
        #print("#",end='')
print ("complete")        