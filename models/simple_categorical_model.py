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
    def set_sigmoid_layers(self):
        self.layers = [
            Layer_Dense(
                n_features=2, n_neurons=1, activation_fn=ActivationFunctions.SIGMOID
            ),
        ]
        self.loss_fn = Loss_MeanSquaredError()

    def set_softmax_layers(self):
        self.layers = [
            Layer_Dense(
                n_features=self.n_features,
                n_neurons=self.n_hidden,
                activation_fn=ActivationFunctions.RELU,
            ),
            Layer_Dense(
                n_features=self.n_hidden,
                n_neurons=self.n_classes,
                activation_fn=ActivationFunctions.SOFTMAX,
            ),
        ]
        self.loss_fn = CategoricalCrossEntropy()

    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        # self.set_sigmoid_layers()
        self.set_softmax_layers()

        super().__init__(self.layers, loss_fn=self.loss_fn)
