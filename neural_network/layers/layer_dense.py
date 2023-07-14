import numpy as np
from numpy.typing import NDArray

from neural_network.activation_functions import activation_fn_map
from neural_network.layers.layer_base import Layer


class Layer_Dense(Layer):
    def __init__(
        self,
        n_features: int,
        n_neurons: int,
        random_state=None,
        activation_fn: str = None,
    ):
        if random_state:
            np.random.seed(random_state)
        self.weights = 0.01 * np.random.randn(n_features, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        if activation_fn:
            if activation_fn in activation_fn_map:
                self.activation_fn = activation_fn_map[activation_fn]()
            else:
                raise ValueError(f"Activation function {activation_fn} not found")

    def set_weights_biases(self, weights: NDArray, biases: NDArray):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

        if hasattr(self, "activation_fn"):
            self.output = self.activation_fn.forward(self.output)

        return self.output

    def backward(self, gradients: NDArray):
        if hasattr(self, "activation_fn"):
            gradients = self.activation_fn.backward(gradients)

        self.weights_gradients = np.dot(self.inputs.T, gradients).reshape(
            self.weights.shape
        )
        self.biases_gradients = np.sum(gradients, axis=0, keepdims=True)

    def update(self, lr):
        self.weights -= lr * self.weights_gradients
        self.biases -= lr * self.biases_gradients

    def __str__(self) -> str:
        if sum(self.weights.shape) > 5 or sum(self.biases.shape) > 3:
            return f"Dense Layer: shape {self.weights.shape}"

        return f"Dense Layer: {self.weights.flatten()}, {self.biases.flatten()}"
