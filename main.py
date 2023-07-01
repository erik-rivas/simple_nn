from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.neural_network import NeuralNetwork


def test_train(features, labels):
    dense = Layer_Dense(n_inputs=2, n_neurons=1)
    layers = [
        dense,
        Activation_Sigmoid(),
    ]

    nn = NeuralNetwork(layers=layers, learning_rate=0.1)

    features = np.array(features)
    labels = np.array(labels)

    nn.train(X=features, y_true=labels, epochs=1000, print_every=100)

    res = nn.forward(features)

    return res


if __name__ == "__main__":
    features = [
        [8, 10],
        [2, -10],
    ]
    labels = [
        [1.0],
        [0.0],
    ]
    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

    features = data[0]
    labels = data[1]

    res = test_train(features, labels)

    x = np.linspace(0, 11, 10)
    y = -x + 5

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Train neural network')

    ax1.plot(x, y)
    ax1.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
    ax1.set_ylabel('Y True')

    ax2.plot(x, y)
    ax2.scatter(features[:, 0], features[:, 1], c=res, cmap="PiYG")
    ax2.set_ylabel('Y True')

    plt.show()

