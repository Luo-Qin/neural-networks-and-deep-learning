import numpy as np

layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

print('sum without axis:')
print(np.sum(layer_outputs))
print(np.sum(layer_outputs, axis=None))

print('sum with axis=0:')
print(np.sum(layer_outputs, axis=0))

print('sum with axis=1:')
print(np.sum(layer_outputs, axis=1))

print('sum with axis=1, but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))
