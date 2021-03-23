import numpy as np

# dense layer
class Layer_Dense:

    # layer initialization 
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # remember input values
        self.inputs = inputs
        # calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass, need to clarify dvalues
    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

