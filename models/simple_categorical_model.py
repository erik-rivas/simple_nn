from matplotlib import pyplot as plt

from libs.idx import read_idx
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.neural_network import NeuralNetwork


class SimpleClassificationModel(NeuralNetwork):
    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = [
            Layer_Dense(
                n_features=n_features, n_neurons=n_hidden, activation_fn="relu"
            ),
            Layer_Dense(
                n_features=n_hidden, n_neurons=n_classes, activation_fn="softmax"
            ),
        ]
        self.loss_fn = CategoricalCrossEntropy()

        super().__init__(self.layers, loss_fn=self.loss_fn)
