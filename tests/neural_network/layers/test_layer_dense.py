import numpy as np
from nptyping import NDArray

from neural_network.layers.layer_dense import Layer_Dense


class TestLayer_Dense:
    def test_init(self):
        layer = Layer_Dense(2, 3, random_state=1)

        assert layer.weights.shape == (2, 3)
        assert layer.biases.shape == (1, 3)

    def test_forward(self):
        layer = Layer_Dense(2, 3, random_state=1)
        inputs = np.ones((1, 2))

        output = layer.forward(inputs)

        assert output.shape == (1, 3)
        assert layer.output.shape == (1, 3)
        assert layer.inputs.shape == (1, 2)

    def test_backward(self):
        layer = Layer_Dense(2, 3, random_state=1)
        inputs = np.ones((1, 2))
        layer.forward(inputs)
        gradients = np.ones((1, 3))

        layer.backward(gradients)

        assert layer.weights_gradients.shape == (2, 3)
        assert layer.biases_gradients.shape == (1, 3)
