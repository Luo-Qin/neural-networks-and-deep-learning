import numpy as np

# ReLU activation 
class Activation_ReLU:

    # forward pass
    def forward(self, inputs):
        # remember input values
        self.inputs = inputs
        # calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        # since we need to modify original variable, 
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

