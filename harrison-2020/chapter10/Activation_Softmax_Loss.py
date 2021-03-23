import numpy as np
import Activation_Softmax as Softmax
import Loss

# softmax classifier - combined softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax.Activation_Softmax()
        self.loss = Loss.Loss_CategoricalCrossentropy()

    # forward pass
    def forward(self, inputs, y_true):
        # output layer's activation function 
        self.activation.forward(inputs)

        # set the output
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # if labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can safely modify
        self.dinputs = dvalues.copy()

        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient 
        self.dinputs = self.dinputs / samples

