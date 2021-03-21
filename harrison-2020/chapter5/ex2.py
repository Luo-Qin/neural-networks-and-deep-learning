import numpy as np

# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# indices of the softmax output distribution
#class_targets = np.array([0, 1, 1])
class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

# probabilities for target values - 
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)), 
        class_targets]

# mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets, axis=1)

print(correct_confidences)

# losses
neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)
print(average_loss)
