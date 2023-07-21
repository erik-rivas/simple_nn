import numpy as np
from matplotlib import pyplot as plt

from libs.idx import read_idx
from neural_network.activation_functions import ActivationFunctions
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.layers.layer_reshape import Layer_Reshape
from neural_network.layers.max_pool import Layer_MaxPool2D
from neural_network.layers.temp_layer_conv2d import Layer_Conv2D
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.neural_network import NeuralNetwork


class MnistModelConv(NeuralNetwork):
    def setup_layers(self):
        self.layers = [
            # Input shape: (batch_size, 1, 28, 28) -> Output shape: (batch_size, 1, 28, 28)
            Layer_Conv2D(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            # Input shape: (1, 28, 28) -> Output shape: (1, 14, 14)
            Layer_MaxPool2D(pool_size=2, stride=2),
            # Input shape (1, 14, 14) -> Output shape: (1, 196)
            Layer_Reshape(shape=(-1, 14 * 14)),
            # Input shape: (1, 196) -> Output shape: (1, 10)
            Layer_Dense(
                n_features=196,
                n_neurons=10,
                activation_fn=ActivationFunctions.RELU,
            ),
            # Input shape: (1, 196) -> Output shape: (1, 10)
            Layer_Dense(
                n_features=10,
                n_neurons=self.n_classes,
                activation_fn=ActivationFunctions.SOFTMAX,
            ),
        ]
        self.loss_fn = CategoricalCrossEntropy()

    def __init__(self, random_state=101):
        np.random.seed(random_state)
        self.n_classes = 10

        self.setup_layers()

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
