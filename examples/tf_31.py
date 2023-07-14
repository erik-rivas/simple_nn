import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.neural_network import NeuralNetwork


def run():
    x_data = np.linspace(0, 10, 1000000)
    noise = np.random.randn(len(x_data))

    m = 0.5
    b = 5

    y_true = (m * x_data) + b + noise

    print(x_data.shape, noise.shape, y_true.shape)

    x_df = pd.DataFrame(x_data, columns=["X Data"])
    y_df = pd.DataFrame(y_true, columns=["Y"])

    my_data = pd.concat([x_df, y_df], axis=1)

    bach_size = 8
    n_baches = 128

    nn = NeuralNetwork(
        layers=[
            Layer_Dense(n_features=1, n_neurons=1),
        ]
    )

    for _ in range(n_baches):
        rand_ind = np.random.randint(len(x_data), size=bach_size)
        x_sample = x_data[rand_ind].reshape((-1, 1))
        y_sample = y_true[rand_ind].reshape(-1, 1)
        nn.train(X=x_sample, y_true=y_sample, learning_rate=0.01, verbose=1000)

    my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")

    print(nn.layers[0])
