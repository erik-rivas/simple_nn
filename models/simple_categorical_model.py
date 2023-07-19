from typing import List

from matplotlib import pyplot as plt

from libs.idx import read_idx
from neural_network.activation_functions import ActivationFunctions
from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError
from neural_network.neural_network import NeuralNetwork


class SimpleClassificationModel(NeuralNetwork):
    def set_layers(self, str_layers: str):
        str_layers = str_layers.split(",")

        layers = []
        for str_layer in str_layers:
            layer = Layer_Dense.from_str(str_layer)
            layers.append(layer)

        self.layers = layers

    def __init__(self, str_layers):
        self.set_layers(str_layers)
        self.loss_fn = CategoricalCrossEntropy()

        super().__init__(self.layers, loss_fn=self.loss_fn)
