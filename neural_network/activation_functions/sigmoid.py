from matplotlib import pyplot as plt
import numpy as np

from neural_network.activation_functions import Activation


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class Activation_Sigmoid(Activation):
    def forward(self, inputs):
        self.output = sigmoid(inputs)
        self.inputs = inputs

        return self.output

    def backward(self, dvalues):
        self.doutput = sigmoid_derivative(dvalues)
        self.dvalues = dvalues

        return self.doutput

    def update(self, learning_rate=None):
        pass

if __name__ == "__main__":
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.show()
