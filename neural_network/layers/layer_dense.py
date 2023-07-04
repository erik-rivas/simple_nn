import numpy as np
from numpy.typing import NDArray

from neural_network.layers.layer_base import Layer


class Layer_Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def set_weights_biases(self, weights: NDArray, biases: NDArray):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

        return self.output

    def backward(self, gradients: np.ndarray):
        self.weights_gradients = np.dot(self.inputs.T, gradients).reshape(
            self.weights.shape
        )
        self.biases_gradients = np.sum(gradients, axis=0, keepdims=True)

    def update(self, lr):
        self.weights -= lr * self.weights_gradients
        self.biases -= lr * self.biases_gradients

    def __str__(self) -> str:
        return f"{self.weights.flatten()}, {self.biases.flatten()}"
