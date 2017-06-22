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



### creating test result

with open ('binary.txt', 'rb') as fp:
    inputList = pickle.load(fp)    


         
mynet = handdigit.network([784,100,10])
mynet.load("MNIST-CrossEntropy-Network")

# Choose a random entry from the test-data.
imgnr = np.random.randint(0,1000)
# Feed it trough the network to get our prediction
print(len(handdigit.testSet[0][imgnr]))


print('test test')


print (handdigit.testSet[0][imgnr])

'''
prediction = mynet.feedforward( handdigit.testSet[0][imgnr] )
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, handdigit.testSet[1][imgnr], np.argmax(prediction)))
'''
prediction = mynet.feedforward( inputList )
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, handdigit.testSet[1][imgnr], np.argmax(prediction)))


# Show the image together with the output neurons
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(inputList, (28,28) ), cmap=cm.gray )
ax[1].plot( prediction, lw=3 )
ax[1].set_aspect(9)
