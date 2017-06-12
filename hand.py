# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:49:01 2017

@author: Ayushc
"""

import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from pylab import *

#%matplotlib inline
 
np.random.seed(1990)

with gzip.open('mnist.pkl.gz','r') as f:
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')

# We restructure the way the data sets are stored to be compatible with the interface for
#  our neural network. We require the input to our neural network to be tuples (x,y) where
#  x is the input and y is the correct output. Moreover, we will encode the output (a number)
#  between 0 and 9 (included) by using 10 output neurons. The neuron with the highest 
#  activation will be taken as the prediction of the network. So the output number y has to 
#  be represented as a list of 10 numbers, all of them being 0 except for the entry at 
#  the correct digit.
learn_data       = [(train_set[0][i], [1 if j == train_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(train_set[0]))]
test_data        = [(test_set[0][i], [1 if j == test_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(test_set[0]))]
validation_data  = [(valid_set[0][i], [1 if j == valid_set[1][i] else 0 for j in range(10)]) \
                    for i in np.arange(len(valid_set[0]))]


# Define the sigmoid function and its derivative
# Notice that we could define the derivative as x*(1-x) and calling this only with
#  the activation of neurons. That saves us the computation of sigmoid(x), but makes
#  for a slightly less readable code. 
def sigmoid( x ):
    return np.nan_to_num( 1/(1+np.exp(-x)) )
def sigmoid_deriv( x ):
    return sigmoid(x)*(1-sigmoid(x))



# In the following cell, we define two different cost functions. The cost function takes
#  the output of the network, given some input, and compares it to the supplied targets for
#  the corresponding inputs. If the predicted outputs differ from the targets, the cost is
#  chosen to be large. The backpropagation algorithm we will use to train the network (which
#  is basically just an exercise in the chain-rule) is based on partial derivatives of the 
#  cost function with repspect to the weights/biases of the network. So the cost functions 
#  should come with a derivative. Also, to make our lives a little bit easier, the classes 
#  supply a 'delta' method which simply combines the first two terms in the chain-rule for 
#  the backpropagation algorithm.

class QuadraticCost:
    """ Cost functions for quadratic cost. """
    
    @staticmethod
    def fn(activations, targets):
        """ Evaluate quadratic cost. """
        return 0.5*(activations - targets)**2

    @staticmethod
    def fn_deriv(activations, targets):
        """ Evaluate derivative of quadratic cost. """
        return activations - targets

    @staticmethod
    def delta(inputs, activations, targets):
        """ Compute the delta error at the output layer for the quadratic cost. """
        return (activations - targets)*sigmoid_deriv(inputs)
    
class CrossEntropyCost:
    """ Cost functions for cross entropy cost. """
    
    @staticmethod
    def fn(activations, targets):
        """ Evaluate cross entropy cost. """
        # The np.nan_to_num function ensures that np.log(0) evaluates to 0 instead of nan.
        return -np.nan_to_num( targets*np.log( activations ) + \
                                                  (1-targets)*np.log( 1 - activations ) )
    
    @staticmethod
    def fn_deriv(activations, targets):
        """ Evaluate the derivative of the cross entropy cost. """
        return -np.nan_to_num( targets/activations - (1-targets)/(1-activations) )
    
    @staticmethod
    def delta(inputs, activations, targets):
        """ Compute the delta error at the output layer for the cross entropy cost. """
        return (activations-targets)

# We now come to the actual definition of the neural network class. The network only 
#  deals with fully connected layers, i.e. no fancy convolutions and pooling. Also, there
#  is currently no dropout mechanism implemented. If you wish to understand the inner 
#  workings of the following code better, I highly suggest looking at Michael Nielsens 
#  website I mentioned in the beginning. Of course, feel free also to contact me!

class neuralnetwork:
    """ A neural network with a bit more thought. """
    
    def __init__(self, shape, cost=CrossEntropyCost):
        """ Initialize the neural network """
        
        # Store shape of the network
        self.shape = shape
        # Give number of layers it's own variable
        self.number_of_layers = len(shape)
        # Set cost function
        self.cost = cost
        
        # Initialize the weight matrices, rescaling the Gaussian to give each neuron a 
        #  relatively peaked activation
        self.weights = [ np.random.normal(0,1/np.sqrt(shape[i+1]),(shape[i], shape[i+1])) \
                          for i in range(self.number_of_layers-1) ]
        # Initialize the biases for all the layers except for the input layer
        self.biases  = [ np.random.normal(0,1,(shape[i])) \
                          for i in range(1,self.number_of_layers) ]
            
    def feedforward( self, inputdata ):
        """ Feed the inputdata through the network """
        
        # Store inputs and outputs for each of the layers
        self.input_to_layer = {}
        self.output_from_layer = {}
        
        # For the input layer, we don't use any activation function
        self.input_to_layer[0] = inputdata
        self.output_from_layer[0] = np.array(inputdata)
    
        # Feed input through the layers
        for layer in range(1,self.number_of_layers):
            self.input_to_layer[layer]    = np.dot( self.output_from_layer[layer-1], \
                                                self.weights[layer-1] ) + self.biases[layer-1]
            self.output_from_layer[layer] = np.array( sigmoid( self.input_to_layer[layer] ) )
    
        # Return output from last layer
        return self.output_from_layer[self.number_of_layers-1]
    
    def backpropagate( self, targets ):
        """ Propage the error backwards, used for gradient descent """
        self.delta = {}
        self.del_cost_del_bias = {}
        self.del_cost_del_weight = {}
        
        # Delta in the final output
        self.delta[self.number_of_layers-1] = \
            (self.cost).delta(self.input_to_layer[self.number_of_layers-1], \
            self.output_from_layer[self.number_of_layers-1], targets )
        
        # Compute the delta's for the other layers
        for layer in np.arange(self.number_of_layers-2, -1, -1):          
            self.delta[layer] = np.dot( self.delta[layer+1],  self.weights[layer].T ) * \
                                        sigmoid_deriv( np.array(self.input_to_layer[layer]) )
            
        # Compute partial derivatives of C w.r.t the biases and the weights
        for layer in np.arange(self.number_of_layers-1, 0, -1):                      
            self.del_cost_del_bias[layer]   = self.delta[layer]
            self.del_cost_del_weight[layer] = np.dot( self.output_from_layer[layer-1].T, \
                                                      self.delta[layer] )
            
        return self.del_cost_del_bias, self.del_cost_del_weight
    
    def train_mini_batch( self, data, rate, l2 ):
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
        for layer in np.arange(1,self.number_of_layers):
            self.biases[layer-1]  -= (rate)*np.mean(self.del_cost_del_bias[layer], axis=0)
            self.weights[layer-1] -= (rate/n)*self.del_cost_del_weight[layer] - \
                                     rate*l2*self.weights[layer-1]
    
    def stochastic_gradient_descent( self, data, number_of_epochs, mini_batch_size, \
                                           rate = 1, l2 = 0.1, test_data = None ):
        """ Train the network using the stochastic gradient descent method. """
        
        # For every epoch:
        for epoch in np.arange(number_of_epochs):
            # Randomly split the data into mini_batches
            np.random.shuffle(data)
            batches = [ data[x:x+mini_batch_size] \
                        for x in np.arange(0, len(data), mini_batch_size) ]
            
            for batch in batches:
                self.train_mini_batch( batch, rate, l2 )
                
            if test_data != None:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), \
                                                           len(test_data)))
                
    def evaluate(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
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
        
mynet = neuralnetwork( [784,100,10] )
mynet.stochastic_gradient_descent( learn_data, 25, 10, 0.1, 0.001/len(train_set[0]), \
                                   test_data = validation_data )        



# saving the current weight and biases
mynet.save("MNIST-CrossEntropy-Network")


# Let's just look at the predictions of our network by showing the
#  input image, printing the actual digit according to MNIST and including
#  a plot of the output neurons of our network. Try running this many times
#  and see on what kinds of digits the network does badly (usually 0 vs 6 or 3 vs 5 and 8)
mynet = neuralnetwork( [784,30,10] )
mynet.load("MNIST-CrossEntropy-Network")

# Choose a random entry from the test-data.
imgnr = np.random.randint(0,10000)
# Feed it trough the network to get our prediction
prediction = mynet.feedforward( test_set[0][imgnr] )
print("Image number {0} is a {1}, and our network predicted a {2}".format(imgnr, test_set[1][imgnr], np.argmax(prediction)))

# Show the image together with the output neurons
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].matshow( np.reshape(test_set[0][imgnr], (28,28) ), cmap=cm.gray )
ax[1].plot( prediction, lw=3 )
ax[1].set_aspect(9)



