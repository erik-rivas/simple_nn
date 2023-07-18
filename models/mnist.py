import numpy as np
from matplotlib import pyplot as plt

from libs.idx import read_idx
from neural_network.activation_functions import ActivationFunctions
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.neural_network import NeuralNetwork


class MnistModel(NeuralNetwork):
    def setup_layers(self, no_layers=2):
        self.n_features = 784
        self.n_hidden = 128
        self.n_classes = 10

        if no_layers == 2:
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
        elif no_layers == 1:
            self.layers = [
                Layer_Dense(
                    n_features=self.n_features,
                    n_neurons=self.n_classes,
                    activation_fn=ActivationFunctions.SOFTMAX,
                ),
            ]
        self.loss_fn = CategoricalCrossEntropy()

    def __init__(self, random_state=101):
        np.random.seed(random_state)

        self.setup_layers(no_layers=2)

        super().__init__(self.layers, loss_fn=self.loss_fn)

    @staticmethod
    def get_mnist(items_to_read=10):
        X_train, _ = read_idx(
            path="data/mnist/train-images-idx3-ubyte", items_to_read=items_to_read
        )

        y_train_onehot, _ = read_idx(
            "data/mnist/train-labels-idx1-ubyte", items_to_read=items_to_read
        )

        X_test, _ = read_idx(
            path="data/mnist/t10k-images-idx3-ubyte", items_to_read=items_to_read
        )

        y_test, _ = read_idx(
            "data/mnist/t10k-labels-idx1-ubyte", items_to_read=items_to_read
        )

        return X_train, y_train_onehot, X_test, y_test

    def plot_dataset_sample(self, sample, label):
        plt.imshow(sample.reshape((28, 28)), cmap="gray")
        plt.title(label)
        plt.show()
