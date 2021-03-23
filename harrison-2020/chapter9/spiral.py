import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import Layer_Dense as ld
import Activation_ReLU as ReLU
import Activation_Softmax_Loss as ASL

nnfs.init()

# create dataset
x, y = spiral_data(samples=100, classes=3)

# create dense layer with 2 input features and 3 output values
dense1 = ld.Layer_Dense(2, 3)

# create ReLU activation (to be used with dense layer):
activation1 = ReLU.Activation_ReLU()

# create second dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = ld.Layer_Dense(3, 3)

# create softmax classifier's combined loss and activation
loss_activation = ASL.Activation_Softmax_Loss_CategoricalCrossentropy()

# perform a forward pass of our training data through this layer
dense1.forward(x)

# perform a forward pass through activation function 
# takes the output of first dense layer here
activation1.forward(dense1.output)

# perform a forward pass through second dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# let's see output of the first few samples:
print(loss_activation.output[:5])

# print loss value
print('loss:', loss)

# calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print('acc:', accuracy)

# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

