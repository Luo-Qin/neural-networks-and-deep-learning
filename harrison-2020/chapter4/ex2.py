import numpy as np

# ReLU activation
class activation_ReLU(object):

    # forward pass
    def forward(self, inputs): 
        # calculate output values from inputs
        self.output = np.maximum(0, inputs)

# Create activation
activation1 = activation_ReLU()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# Perform a forward pass of our training data through this layer
activation1.forward(inputs)

output = activation1.output

# Let's see output of the first few samples:
print(output)

