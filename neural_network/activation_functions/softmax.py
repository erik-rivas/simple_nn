import numpy as np

from neural_network.activation_functions.activation import ActivationFunction


class Activation_SoftMax(ActivationFunction):
    """
    Softmax activation function.
    """

    def forward(self, raw_inputs):
        exponents = np.sum(np.exp(raw_inputs), axis=1, keepdims=True)
        probabilities = np.round(np.exp(raw_inputs) / exponents, 3)

        self.output = probabilities
        self.inputs = raw_inputs

        return self.output

    def backward(self, crossentropy_gradient):
        self.dinputs = np.empty_like(crossentropy_gradient)

        for index, (single_output, single_gradient) in enumerate(
            zip(self.output, crossentropy_gradient)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[index] = np.dot(jacobian_matrix, single_gradient)

        return self.dinputs

    def __repr__(self) -> str:
        return "softmax"

    def __str__(self) -> str:
        return f"<Activation Softmax>"
