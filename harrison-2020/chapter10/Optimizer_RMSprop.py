import numpy as np

# RMSprop optimizer
class Optimizer_RMSprop:

    # initialize optimizer - set settings, 
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
 
    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1 / (1 + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):

        # if layer does not contain cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
             (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
             (1 - self.rho) * layer.dbiases**2

        # vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / \
             (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
             (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

