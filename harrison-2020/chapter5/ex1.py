import numpy as np

# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# indices of the softmax output distribution
class_targets = [0, 1, 1]

# map these indices to retrieve the values from the softmax distribution
# print(softmax_outputs[range(len(softmax_outputs)), class_targets])

# apply the negative log to this list
# print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))

# calculate an average loss per batch
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])

average_loss = np.mean(neg_log)
print(average_loss)

