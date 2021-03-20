import numpy as np
import nnfs
import matplotlib.pyplot as plt

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

# ReLU activation
class activation_ReLU(object):

    # forward pass
    def forward(self, inputs): 
        # calculate output values from inputs
        self.output = np.maximum(0, inputs)

# softmax activation
class activation_softmax:

    # forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = layer_dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(x)

# Create ReLU activation for hidden layer
activation1 = activation_ReLU()

# Forward pass through activation func.
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])

# create second dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = layer_dense(3,3 )

# create softmax activation (to be used with dense layer)
activation2 = activation_softmax()

# make a forward pass through second dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# let's see output of the first few samples:
print(activation2.output[:5])

