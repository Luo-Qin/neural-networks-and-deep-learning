import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dense layer
class Layer_Dense:

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
class Activation_ReLU:

    # forward pass
    def forward(self, inputs): 
        # calculate output values from inputs
        self.output = np.maximum(0, inputs)


# softmax activation
class Activation_Softmax:

    # forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, 
            keepdims=True))

        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
            keepdims=True)

        self.output = probabilities


# common loss class
class Loss: 

    # calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # calculate sample losses
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # return loss
        return data_loss


# cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # number of samples in a batch
        samples = len(y_pred)

        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true]

        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis=1)

        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation for hidden layer
activation1 = Activation_ReLU()

# create second dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# create softmax activation (to be used with dense layer)
activation2 = Activation_Softmax()

# create loss function
loss_function = Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(x)

# Forward pass through activation func.
activation1.forward(dense1.output)

# make a forward pass through second dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# let's see output of the first few samples:
print(activation2.output[:5])

# perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# print loss value
print('loss:', loss)

