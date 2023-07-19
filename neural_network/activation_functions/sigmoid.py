import numpy as np
from matplotlib import pyplot as plt

from neural_network.activation_functions.activation import ActivationFunction


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(z_sigmoid):
    # compute the derivative of the sigmoid function ASSUMING
    # that x has already been passed through the 'sigmoid'
    # function
    return z_sigmoid * (1 - z_sigmoid)

    # den = (1 + np.exp(-z)) ** 2
    # return np.exp(-z) / den


class Activation_Sigmoid(ActivationFunction):
    def forward(self, inputs):
        self.output = sigmoid(inputs)
        self.inputs = inputs

        return self.output

    def backward(self, dvalues):
        self.doutput = sigmoid_derivative(dvalues)
        self.dvalues = dvalues

        return self.doutput

    def __repr__(self) -> str:
        return "sigmoid"

    def __str__(self) -> str:
        return f"<Activation Sigmoid>"


if __name__ == "__main__":
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    dy_dx = sigmoid_derivative(y)

    ax1 = plt.subplot(211)
    ax1.plot(x, y)

    ax2 = plt.subplot(212, sharex=ax1)
    ax1.plot(x, dy_dx)

    plt.show()
