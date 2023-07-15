import numpy as np
import pytest

from neural_network.activation_functions.softmax import Activation_SoftMax


class TestSoftMax:
    @pytest.mark.parametrize(
        # fmt: off
        "raw_inputs, expected",
        [
            (
                np.array(
                    [
                        [0, 1],
                    ]
                ),
                np.array(
                    [
                        [0.268, 0.731],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ]
                ),
                np.array(
                    [
                        [0.090, 0.244, 0.665],
                        [0.090, 0.244, 0.665]
                    ]
                ),
            ),
        ],
        # fmt: on
    )
    def test_softmax(self, raw_inputs, expected):
        layer = Activation_SoftMax()
        output = layer.forward(raw_inputs)

        assert np.allclose(output, expected, atol=0.001)
