# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:10:16 2017

@author: ayushc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:07:10 2017

@author: Ayushc
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import alphabethand
import pickle



### creating test result

with open ('Hnd/binarychara2.txt', 'rb') as fp:
    inputList = pickle.load(fp)    


         
mynet =alphabethand.network([784,100,26])
mynet.load("MNIST-CrossEntropy-Network-Char")

# Choose a random entry from the test-data.
imgnr = np.random.randint(0,1000)
# Feed it trough the network to get our prediction
#print(len(alphabethand.itemlist[1020][0] ))


print('inputvslur')


print (inputList)

print('3###############')
print(inputList)

prediction = mynet.feedforward(inputList)
#prediction = mynet.feedforward( inputList )
print("predictoiojskjfs")
print(prediction)
ans=np.argmax(prediction)
print("ans safdas")
print(ans)
print(chr(65+ans))
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr,inputList, chr(65+ans)))


#prediction = mynet.feedforward( alphabethand.itemlist[1020][0] )
#print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, alphabethand.itemlist[1020][0], np.argmax(prediction)))



# Show the image together with the output neurons
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape( inputList, (28,28) ), cmap=cm.gray )
ax[1].plot( prediction, lw=3 )
ax[1].set_aspect(9)