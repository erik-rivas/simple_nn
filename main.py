import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from examples import tf_31, tf_32
from libs.helpers import spiral_data
from neural_network.activation_functions import (
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_Softmax,
)
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.layer_crossentropy import (
    Loss_CategoricalCrossentropy_Loss,
)
from neural_network.neural_network import NeuralNetwork


def test_simple_regression():
    x = np.linspace(0, 10, 101).reshape((-1, 1))
    y_true = x * 0.5 + 10 + np.random.uniform(-2, 2)
    print(y_true)

    layers = [
        Layer_Dense(n_features=1, n_neurons=1),
        # Activation_Sigmoid(),
    ]

    nn = NeuralNetwork(layers=layers)

    nn.train(X=x, y_true=y_true, learning_rate=0.01, epochs=1000, print_every=100)

    x_pred = np.linspace(0, 10, 11).reshape((-1, 1))
    y_pred = nn.forward(x_pred)

    plt.plot(x, y_true, c="blue")
    plt.plot(x_pred, y_pred, "o", c="red")

    plt.show()


def test_linnear_classifier():
    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
    features = data[0]
    labels = data[1].reshape(-1, 1)

    dense = Layer_Dense(n_features=2, n_neurons=1)
    layers = [
        dense,
        Activation_Sigmoid(),
    ]

    nn = NeuralNetwork(layers=layers)

    features = np.array(features)
    labels = np.array(labels)

    nn.train(
        X=features, y_true=labels, learning_rate=0.01, epochs=1000, print_every=100
    )

    y_predicted = nn.forward(features)

    x = np.linspace(0, 11, 10)
    y = -x + 5

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Train neural network")

    ax1.plot(x, y)
    ax1.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
    ax1.set_ylabel("Y True")

    ax2.plot(x, y)
    ax2.scatter(features[:, 0], features[:, 1], c=y_predicted, cmap="PiYG")
    ax2.set_ylabel("Y Predicted")

    plt.show()


def test_spiral():
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

    nn = NeuralNetwork(layers=layers)
    nn.train(
        X=features, y_true=labels, learning_rate=0.01, epochs=1000, print_every=100
    )
    res = nn.forward(features)


if __name__ == "__main__":
    test_simple_regression()
    test_linnear_classifier()
    # test_spiral()

    tf_31.run()
    tf_32.run()
