from typing import List

from neural_network.activation_functions import ActivationFunctions
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.neural_network import NeuralNetwork


class SimpleSpiralModel(NeuralNetwork):
    def set_layers(self, str_layers: List[str] = None):
        layers = [
            Layer_Dense(
                n_features=self.n_features,
                n_neurons=10,
                activation_fn=ActivationFunctions.TANH,
            ),
            Layer_Dense(
                n_features=10,
                n_neurons=10,
                activation_fn=ActivationFunctions.TANH,
            ),
            Layer_Dense(
                n_features=10,
                n_neurons=self.n_classes,
                activation_fn=ActivationFunctions.SOFTMAX,
            ),
        ]
        print("Layers:")
        for layer in layers:
            print(layer)

        str_layers = str_layers or [
            f"{self.n_features}::10_tanh",
            "10::10_tanh",
            f"10::{self.n_classes}_softmax",
        ]
        layers = []
        for str_layer in str_layers:
            layer = Layer_Dense.from_str(str_layer)
            layers.append(layer)

        print("Layers 2:")
        for layer in layers:
            print(layer)
        self.layers = layers
        self.loss_fn = CategoricalCrossEntropy()

    def __init__(self, n_classes, str_layers: List[str] = []):
        """
        Initialize a simple neural network with 2 hidden layers and tanh activation function.
        str_layers: list of strings, each string is a layer type, e.g. "dense", "conv", "pool", "flatten"
        """
        self.n_features = 2
        self.n_classes = n_classes

        self.set_layers(str_layers)

        super().__init__(self.layers, loss_fn=self.loss_fn)
