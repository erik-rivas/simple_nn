import numpy as np
from numpy.typing import NDArray

from neural_network.activation_functions import ActivationFunctions, activation_fn_map
from neural_network.layers.layer_base import Layer


class Layer_Dense(Layer):
    def __init__(
        self,
        n_features: int,
        n_neurons: int,
        random_state: int = None,
        activation_fn: ActivationFunctions = None,
    ):
        self.n_features = n_features
        self.n_neurons = n_neurons

        if random_state:
            np.random.seed(random_state)
        self.weights = 0.01 * np.random.randn(n_features, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        if activation_fn:
            if activation_fn in activation_fn_map:
                self.activation_fn = activation_fn_map[activation_fn]()
            else:
                raise ValueError(f"Activation function {activation_fn} not found")

    @classmethod
    def from_str(cls, str_layer: str):
        """
        str_layer: string, e.g. "2::10_relu"
        """
        if "::" not in str_layer or "_" not in str_layer:
            raise ValueError("Invalid string layer")

        # split string into n_features, n_neurons, activation_fn
        (temp, str_activation_fn) = str_layer.split("_")
        (str_n_features, str_n_neurons) = temp.split("::")

        # parse strings
        n_features = int(str_n_features)
        n_neurons = int(str_n_neurons)
        activation_fn = ActivationFunctions(str_activation_fn)

        dense_layer = cls(n_features, n_neurons, activation_fn=activation_fn)
        print(dense_layer)

        return dense_layer

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
        self.dinputs = np.dot(gradients, self.weights.T)

        return self.dinputs

    def update(self, lr):
        self.weights -= lr * self.weights_gradients
        self.biases -= lr * self.biases_gradients

    def __str__(self) -> str:
        return (
            f"||DenseLayer: {self.n_features}::{self.n_neurons}_{self.activation_fn} ||"
        )
