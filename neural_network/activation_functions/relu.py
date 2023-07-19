import numpy as np

from neural_network.activation_functions.activation import ActivationFunction


class Activation_ReLU(ActivationFunction):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs

    def __repr__(self) -> str:
        return "relu"

    def __str__(self) -> str:
        return f"<Activation ReLU>"
