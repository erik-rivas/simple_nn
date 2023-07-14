import numpy as np

from neural_network.activation_functions.activation import ActivationFunction


class Activation_SoftMax(ActivationFunction):
    """
    Softmax activation function.
    """

    def forward(self, inputs):
        exponents = sum(np.exp(inputs))
        probabilities = np.round(np.exp(inputs) / exponents, 3)

        self.output = probabilities
        self.inputs = inputs

        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def __str__(self) -> str:
        return f"Activation Softmax"
