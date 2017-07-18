# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:52:42 2017

@author: Ayushc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:26:19 2017

@author: Ayushc
"""

import numpy as np
import pickle
import gzip
import cv2

np.random.seed(1990)

with gzip.open('mnist.pkl.gz','r') as f:
     trainSet, validSet, testSet = pickle.load(f,encoding='latin1')
     
####################################
''' generating the train set'''
b=[]
singleLearndata=[]
a=[0,0,0,0,0,1,0,0,0,0]
img= cv2.imread('sampleimg.png',0)
img=cv2.resize(img,(28,28))
ret,thresh=cv2.threshold(img,150,1,cv2.THRESH_BINARY_INV)
b.append(np.float32(thresh.ravel()))
b.append(a)
####################################     
#print ('##train set data is here hello where  are you man what enough ########')
#print (len(trainSet[0]))     

singleLearndata.append(tuple(b))
print('b',b)
print('single learn data',singleLearndata[0])    
'''     
learnData = [(trainSet[0][i], [1 if j == trainSet[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(trainSet[0]))]   
print ("learn data 1 ")
print(learnData[0])

#print(learnData)
     
#print (learnData)
'''
testData        = [(testSet[0][i], [1 if j == testSet[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(testSet[0]))]
#print (len(testData[0][0]))
validationData  = [(validSet[0][i], [1 if j == validSet[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(validSet[0]))]     



#print(validationData)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    

class QuadraticCost:
    """ Cost functions for quadratic cost. """
    
    @staticmethod
    def fn(activations, targets):
        """ Evaluate quadratic cost. """
        return 0.5*(activations - targets)**2

    @staticmethod
    def fnDeriv(activations, targets):
        """ Evaluate derivative of quadratic cost. """
        return activations - targets

    @staticmethod
    def delta(inputs, activations, targets):
        """ Compute the delta error at the output layer for the quadratic cost. """
        return (activations - targets)*dSigmoid(inputs)
    
class CrossEntropyCost:
    """ Cost functions for cross entropy cost. """
    
    @staticmethod
    def fn(activations, targets):
        """ Evaluate cross entropy cost. """
        # The np.nan_to_num function ensures that np.log(0) evaluates to 0 instead of nan.
        return -np.nan_to_num( targets*np.log( activations ) + \
                                                  (1-targets)*np.log( 1 - activations ) )
    
    @staticmethod
    def fnDeriv(activations, targets):
        """ Evaluate the derivative of the cross entropy cost. """
        return -np.nan_to_num( targets/activations - (1-targets)/(1-activations) )
    
    @staticmethod
    def delta(inputs, activations, targets):
        """ Compute the delta error at the output layer for the cross entropy cost. """
        return (activations-targets)








class networkself:
    def __init__(self,shape,cost=CrossEntropyCost):
        
        
        
         # Store shape of the network
        self.shape = shape
        # Give number of layers it's own variable
        self.numberOfLayers = len(shape)
        # Set cost function
        self.cost = cost
        
        # Initialize the weight matrices, rescaling the Gaussian to give each neuron a 
        #  relatively peaked activation
        self.weights = [ np.random.normal(0,1/np.sqrt(shape[i+1]),(shape[i], shape[i+1])) \
                          for i in range(self.numberOfLayers-1) ]
        # Initialize the biases for all the layers except for the input layer
        self.biases  = [ np.random.normal(0,1,(shape[i])) \
                          for i in range(1,self.numberOfLayers) ]
        
    

    def feedforward(self,inputData):
        
        self.layerInput={}
        self.layerOutput={}
        
        self.layerInput[0]=inputData
        self.layerOutput[0]=np.array(inputData)
        
        
           # Feed input through the layers
        for layer in range(1,self.numberOfLayers):
            self.layerInput[layer]    = np.dot( self.layerOutput[layer-1], \
                                                self.weights[layer-1] ) + self.biases[layer-1]
            self.layerOutput[layer] = np.array( sigmoid( self.layerInput[layer] ) )
            
   # Return output from last layer
        #print("feed return value")
        #print(self.layerOutput[self.numberOfLayers-1])
        return self.layerOutput[self.numberOfLayers-1]           
            
            
    def backpropagate( self, targets ):
        """ Propage the error backwards, used for gradient descent """
        self.delta = {}
        self.delCostBias = {}
        self.delCostWeight = {}
        
        # Delta in the final output
        self.delta[self.numberOfLayers-1] = \
            (self.cost).delta(self.layerInput[self.numberOfLayers-1], \
            self.layerOutput[self.numberOfLayers-1], targets )
        
        # Compute the delta's for the other layers
        for layer in np.arange(self.numberOfLayers-2, -1, -1):          
            self.delta[layer] = np.dot( self.delta[layer+1],  self.weights[layer].T ) * \
                                        dSigmoid( np.array(self.layerInput[layer]) )
            
        # Compute partial derivatives of C w.r.t the biases and the weights
        for layer in np.arange(self.numberOfLayers-1, 0, -1):                      
            self.delCostBias[layer]   = self.delta[layer]
            self.delCostWeight[layer] = np.dot( self.layerOutput[layer-1].T, \
                                                      self.delta[layer] )
            
        return self.delCostBias, self.delCostWeight



    def trainMiniBatch( self, data, rate, l2 ):
        
        """ Train the network on a mini-batch """
        
            # Split the data into input and output
        inputs  = [ entry[0] for entry in data ]
        targets = [ entry[1] for entry in data ]
            
        # Feed the input through the network
        self.feedforward( inputs )
        # Propagate the error backwards
        self.backpropagate( targets )
            
            # Update the weights and biases
        n = len(targets)
        for layer in np.arange(1,self.numberOfLayers):
             self.biases[layer-1]  -= (rate)*np.mean(self.delCostBias[layer], axis=0)
             self.weights[layer-1] -= (rate/n)*self.delCostWeight[layer] - \
                                         rate*l2*self.weights[layer-1]
        
 ###training using stochist gradient method
    
    def stochasticGradientDescent( self, data, numberOfEpochs, miniBatchSize, \
                                               rate = 1, l2 = 0.1, testData = None ):
            """ Train the network using the stochastic gradient descent method. """
            
            # For every epoch:
            for epoch in np.arange(numberOfEpochs):
                # Randomly split the data into mini_batches
                np.random.shuffle(data)
                batches = [ data[x:x+miniBatchSize] \
                            for x in np.arange(0, len(data), miniBatchSize) ]
                
                for batch in batches:
                    self.trainMiniBatch( batch, rate, l2 )
                    
                if testData != None:
                    print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(testData), \
                                                               len(testData)))
        
            
    def evaluate(self, test_data):
            """ Evaluate performance by counting how many examples in tesData are correctly 
                evaluated. """
            count = 0
            for testcase in test_data:
                #print ("test case0")
                #print(testcase[0])
                #print ("tesetcase 1")
                #print(testcase[1])
                answer = np.argmax( testcase[1] )
                #print ("answer")
                #print (answer)
                prediction = np.argmax( self.feedforward( testcase[0] ) )
                count = count + 1 if (answer - prediction) == 0 else count
            return count     
                
    
    def save(self, filename):
            """ Save neural network (weights) to a file. """
            with open(filename, 'wb') as f:
                pickle.dump({'biases':self.biases, 'weights':self.weights}, f )
            
    def load(self, filename):
            """ Load neural network (weights) from a file. """
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Set biases and weights
            self.biases = data['biases']
            self.weights = data['weights']
 

def main():
    print ('In main')
            
    mynet = networkself( [784,100,10] )
    mynet.load("MNIST-CrossEntropy-Network") 
    mynet.stochasticGradientDescent( singleLearndata, 25, 10, 0.1, 0.001/len(trainSet[0]), \
                                   testData = validationData )        



# saving the current weight and biases
    mynet.save("MNIST-CrossEntropy-Network")

if __name__ == '__main__':
    main()
   
        
        
        