import numpy as np

# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

# target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])
# class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

# calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis=1)
print('predictions: ', predictions)

# if targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)
print('class_targets: ', class_targets)

# True evaluates to 1; false to 0
accuracy = np.mean(predictions==class_targets)
print('accuracy: ', accuracy)

