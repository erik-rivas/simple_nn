import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from libs.helpers import spiral_data
from neural_network import NeuralNetwork
from neural_network.activation_functions import (
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_SoftMax,
)
from neural_network.layers import Layer_Dense
from neural_network.loss_functions import CategoricalCrossEntropy


class TestNeuralNetwork:
    def test_nn_line(self):
        data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=101)

        features = data[0]
        labels = data[1]

        x = np.linspace(0, 11, 10)
        y = -x + 5

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        # plt.plot(x, y)
        # plt.show()

        dense = Layer_Dense(n_features=2, n_neurons=1)
        dense.set_weights_biases(weights=np.array([[1], [1]]), biases=np.array([-5]))
        # dense.set_weights_biases(5, 5)

        layers = [dense, Activation_Sigmoid()]

        nn = NeuralNetwork(layers=layers)

        x = np.array([8, 10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.9999, abs=1e-4)

        x = np.array([2, -10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.000002, abs=1e-6)

    def test_nn_line_backward(self):
        data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=101)

        features = data[0]
        labels = data[1].reshape(-1, 1)

        x = np.linspace(0, 11, 10)
        y = -x + 5

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        # plt.plot(x, y)
        # plt.show()

        dense = Layer_Dense(n_features=2, n_neurons=1)

        layers = [dense, Activation_Sigmoid()]

        nn = NeuralNetwork(layers=layers)

        nn.train(
            X=features, y_true=labels, learning_rate=0.01, epochs=1000, verbose=100
        )

        x = np.array([8, 10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.9999, abs=1e-4)

        x = np.array([2, -10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.0001, abs=1e-3)

    def test_spiral(self):
        features, labels = spiral_data(100, 2)

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        # plt.show()

        layers = [
            Layer_Dense(2, 64),
            Activation_ReLU(),
            Layer_Dense(64, 3),
            Activation_SoftMax(),
            CategoricalCrossEntropy(),
        ]
