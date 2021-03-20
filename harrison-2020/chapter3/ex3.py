import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dense layer
class layer_dense(object):

    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, n_inputs): 
        # calculate output values from inputs, weights and biases
        self.output = np.dot(n_inputs, self.weights) + self.biases

# create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = layer_dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])

# question: how to get each element of the output? 

