import numpy as np

# passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])

# we have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2, 5, -5, 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# we have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# one bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# forward pass
# dense layer
layer_outputs = np.dot(inputs, weights) + biases 
# ReLU activation
relu_outputs = np.maximum(0, layer_outputs)

#let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation 
drelu = relu_outputs.copy()
drelu[layer_outputs<=0] = 0

# dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)

