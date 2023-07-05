import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from libs.helpers import spiral_data
from neural_network import NeuralNetwork
from neural_network.activation_functions import Activation_Sigmoid
from neural_network.activation_functions.relu import Activation_ReLU
from neural_network.activation_functions.softmax import Activation_Softmax
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.layer_crossentropy import (
    Loss_CategoricalCrossentropy_Loss,
)


class TestNeuralNetwork:
    def test_nn_line(self):
        data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

        features = data[0]
        labels = data[1]

        x = np.linspace(0, 11, 10)
        y = -x + 5

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        plt.plot(x, y)
        # plt.show()

        dense = Layer_Dense(n_inputs=2, n_neurons=1)
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
        data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

        features = data[0]
        labels = data[1]

        x = np.linspace(0, 11, 10)
        y = -x + 5

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        plt.plot(x, y)
        # plt.show()

        dense = Layer_Dense(n_inputs=2, n_neurons=1)
        # dense.set_weights_biases(weights=np.array([[1], [1]]), biases=np.array([-5]))

        layers = [dense, Activation_Sigmoid()]

        nn = NeuralNetwork(layers=layers)

        nn.train(
            X=features, y_true=labels, learning_rate=0.01, epochs=1000, print_every=100
        )

        x = np.array([8, 10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.9999, abs=1e-4)

        x = np.array([2, -10])
        res = nn.forward(x)

        assert res[0] == pytest.approx(0.00001, abs=1e-5)

    def test_spiral(self):
        features, labels = spiral_data(100, 2)

        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
        plt.show()

        layers = [
            Layer_Dense(2, 64),
            Activation_ReLU(),
            Layer_Dense(64, 3),
            Activation_Softmax(),
            Loss_CategoricalCrossentropy_Loss(),
        ]
