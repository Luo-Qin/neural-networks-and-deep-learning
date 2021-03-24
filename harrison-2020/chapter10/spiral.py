import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import Layer_Dense as ld
import Activation_ReLU as ReLU
import Activation_Softmax_Loss as ASL
import Optimizer_SGD as SGD
import Optimizer_Adagrad as Adagrad
import Optimizer_RMSprop as RMSprop
import Optimizer_Adam as Adam

nnfs.init()

# create dataset
x, y = spiral_data(samples=100, classes=3)

print('x.shape:', x.shape)
print('y.shape:', y.shape)
print('len(y.shape):', len(y.shape))

# create dense layer with 2 input features and 64 output values
dense1 = ld.Layer_Dense(2, 64)

# create ReLU activation (to be used with dense layer):
activation1 = ReLU.Activation_ReLU()

# create second dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = ld.Layer_Dense(64, 3)

# create softmax classifier's combined loss and activation
loss_activation = ASL.Activation_Softmax_Loss_CategoricalCrossentropy()

# create optimizer
#optimizer = SGD.Optimizer_SGD(decay=1e-3, momentum=0.9)
#optimizer = Adagrad.Optimizer_Adagrad(decay=1e-4)
#optimizer = RMSprop.Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam.Optimizer_Adam(learning_rate=0.02, decay=1e-5)

# train in loop
for epoch in range(10001):

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

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ', \
              f'acc: {accuracy:.3f}, ' \
              f'loss: {loss:.3f}, ' \
              f'lr: {optimizer.current_learning_rate}')

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

