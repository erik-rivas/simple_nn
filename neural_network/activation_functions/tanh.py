import numpy as np
from matplotlib import pyplot as plt

from neural_network.activation_functions.activation import ActivationFunction


class Activation_Tanh(ActivationFunction):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs

        return self.output

    def backward(self, dvalues):
        self.doutput = dvalues * (1 - self.output**2)
        self.dvalues = dvalues

        return self.doutput

    def __repr__(self) -> str:
        return "tanh"

    def __str__(self) -> str:
        return f"<Activation Tanh>"


if __name__ == "__main__":
    fn = Activation_Tanh()
    x = np.linspace(-10, 10, 100)
    y = fn.forward(x)
    dy_dx = fn.backward(y)

    ax1 = plt.subplot(211)
    ax1.plot(x, y)

    ax2 = plt.subplot(212, sharex=ax1)
    ax1.plot(x, dy_dx)

    plt.show()
