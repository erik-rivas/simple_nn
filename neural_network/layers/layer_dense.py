import numpy as np


# Then we define our classes
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def set_weights_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

        return self.output

    def backward(self, gradients):
        self.weights_gradients = np.dot(self.inputs.T, gradients)
        self.biases_gradients = np.sum(gradients, axis=0, keepdims=True)
        self.input_gradients = np.dot(gradients, self.weights.T)

    def update(self, lr):
        self.weights -= lr * self.weights_gradients
        self.biases -= lr * self.biases_gradients
