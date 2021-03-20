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

# create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = layer_dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(x)

# Create ReLU activation
activation1 = activation_ReLU()

# Forward pass through activation func.
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])

# save the plot
plt.figure()
plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
plt.savefig("spiral.png")

output = dense1.output
plt.figure()
plt.scatter(output[:,0], output[:,1], c=output[:,2], cmap='brg')
plt.savefig("spiral_dense.png")

output = activation1.output
plt.figure()
plt.scatter(output[:,0], output[:,1], c=output[:,2], cmap='brg')
plt.scatter(output[:,0], output[:,1], c=output[:,2], cmap='brg')
plt.savefig("spiral_ReLU.png")

