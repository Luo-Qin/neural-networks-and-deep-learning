import numpy as np
import network1 as nw
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# create dataset
x, y = vertical_data(samples=100, classes=3)

# create model
dense1 = nw.Layer_Dense(2,3)
activation1 = nw.Activation_ReLU()
dense2 = nw.Layer_Dense(3,3)
activation2 = nw.Activation_Softmax()

# create loss function
loss_function = nw.Loss_CategoricalCrossentropy()
#print(loss_function)

# helper variables 
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):

    # generate a new set of weights for iteration
    dense1.weights = 0.05 * np.random.randn(2,3)
    dense1.biases = 0.05 * np.random.randn(1,3)
    dense2.weights = 0.05 * np.random.randn(3,3)
    dense2.biases = 0.05 * np.random.randn(1,3)

    # perform a forward pass of the training data through this layer
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # perform a forward pass through activation function
    # it takes the output of second dense layer here and return loss
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    # if loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('new set of weights found, iteration:', iteration,
            'loss', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

