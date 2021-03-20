import numpy as np

# values from the previous output when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# for each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

# now normalize values
norm_values = exp_values/np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))
