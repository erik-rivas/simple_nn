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

    features=np.array(features)
    labels=np.array(labels)

    nn.train(X=features, y_true=labels, epochs=1000, print_every=100)

    res = nn.forward(features)

    return res


if __name__ == "__main__":

    features = [
        [8, 10],
        [2, -10],
    ]
    labels = [[1.0], [0.0]]

    res = test_train(features, labels)
    print(res)
