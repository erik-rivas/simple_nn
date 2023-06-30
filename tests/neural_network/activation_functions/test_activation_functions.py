import numpy as np
import pytest

from neural_network.activation_functions.sigmoid import sigmoid


class TestActivationSigmoid:
    def test_sigmoid(self):
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)

        assert y.shape == x.shape
