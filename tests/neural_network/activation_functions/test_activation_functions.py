import numpy as np
import pytest

from neural_network.activation_functions import Activation_Sigmoid


class TestActivationSigmoid:
    def test_sigmoid(self):
        activation = Activation_Sigmoid()
        assert activation.forward(0) == 0.5
        assert activation.backward(0.5) == 0.25
