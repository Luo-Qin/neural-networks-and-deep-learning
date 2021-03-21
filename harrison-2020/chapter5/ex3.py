import numpy as np

# probabilities for 3 samples
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# indices of the softmax output distribution
#class_targets = np.array([0, 1, 1])
class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

# common loss class
class loss: 

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
class loss_categoricalCrossEntropy(loss):

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

loss_function = loss_categoricalCrossEntropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)

