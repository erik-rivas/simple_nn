import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.estimators.linear_regressor import LinearRegressor
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.neural_network import NeuralNetwork


def run():
    x_data = np.linspace(0, 10, 1000000)
    noise = np.random.randn(len(x_data))

    m = 0.5
    b = 5

    y_true = (m * x_data) + b + noise

    estimator = LinearRegressor()

    weights, bias = estimator.train(x_data, y_true, n_baches=100)
    print(f"weights: {weights}, bias: {bias}")

    x_test = np.linspace(0, 10, 10).reshape(-1, 1)
    y_test = estimator.forward(x_test)

    plt.scatter(x_data, y_true)
    plt.plot(x_test, y_test, "r*")
    plt.show()
