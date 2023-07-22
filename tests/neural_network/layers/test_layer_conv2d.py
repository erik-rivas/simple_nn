import numpy as np
import pytest

from neural_network.layers.layer_conv2d import Conv2D


def get_params(extra_args=[]):
    layer_input = np.arange(16).reshape(1, 1, 4, 4)
    args = [
        {
            "input": layer_input,
            "kernel": np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ).reshape(1, 1, 3, 3),
            "bias": np.array([0]).reshape(1, 1, 1, 1),
            "expected_output": np.arange(16).reshape(1, 1, 4, 4),
            "expected_d_input": (np.ones((4, 4)) * 1).reshape(1, 1, 4, 4),
        },
        {
            "input": layer_input,
            "kernel": np.array([[0, 0], [0, 1]]).reshape(1, 1, 2, 2),
            "bias": np.array([0]).reshape(1, 1, 1, 1),
            "expected_output": np.pad(
                np.arange(16).reshape(1, 1, 4, 4),
                [(0, 0), (0, 0), (0, 1), (0, 1)],
                mode="constant",
            ),
            "expected_d_input": np.ones((5, 5)).reshape(1, 1, 5, 5),
        },
    ]

    if "expected_d_filters" in extra_args:
        for arg in args:
            if arg["kernel"].shape[-1] == 2:
                arg["expected_d_filters"] = np.array(
                    [
                        [120.0, 120.0],
                        [120.0, 120.0],
                    ]
                ).reshape(1, 1, 2, 2)
            elif arg["kernel"].shape[-1] == 3:
                arg["expected_d_filters"] = np.array(
                    [
                        [45.0, 66.0, 54.0],
                        [84.0, 120.0, 96.0],
                        [81.0, 114.0, 90.0],
                    ]
                ).reshape(1, 1, 3, 3)

    return [arg.values() for arg in args]


class TestLayerConv2d:
    @pytest.mark.parametrize(
        "input,kernel, bias, expected_output, expected_d_input", get_params()
    )
    def test_forward(self, input, kernel, bias, expected_output, expected_d_input):
        layer = Conv2D(
            in_channels=1, out_channels=1, kernel_size=kernel.shape[2], padding=1
        )
        layer.set_weights_biases(kernel, bias)

        output = layer.forward(input)

        assert np.allclose(output, expected_output, atol=1e-6)

    @pytest.mark.parametrize(
        "input,kernel, bias, expected_output, expected_d_input", get_params()
    )
    def test_backward(self, input, kernel, bias, expected_output, expected_d_input):
        layer = Conv2D(
            in_channels=1, out_channels=1, kernel_size=kernel.shape[-1], padding=1
        )
        layer.set_weights_biases(kernel, bias)

        output = layer.forward(input)

        d_out = np.ones_like(output)
        d_input = layer.backward(d_out=d_out)

        assert np.allclose(output, expected_output, atol=1e-6)
        assert np.allclose(d_input, expected_d_input, atol=1e-6)

    @pytest.mark.parametrize(
        "input, kernel, bias, expected_output, expected_d_input, expected_d_filters",
        get_params(["expected_d_filters"]),
    )
    def test_forward_backward(
        self, input, kernel, bias, expected_output, expected_d_input, expected_d_filters
    ):
        # Initialize Conv2D layer
        conv = Conv2D(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel.shape[-1],
            stride=1,
            padding=1,
        )
        conv.set_weights_biases(kernel, bias)

        # Forward pass
        output = conv.forward(input)

        assert np.allclose(output, expected_output, atol=1e-6)

        # Backward pass
        d_out = np.ones_like(output)
        d_input = conv.backward(d_out)

        # Expected gradients
        assert np.allclose(d_input, expected_d_input, atol=1e-6)
        assert np.allclose(conv.d_filters, expected_d_filters, atol=1e-6)
        # assert np.allclose(conv.d_filters, expected_d_filters, atol=1e-6)

        # assert np.allclose(
        #     conv.d_filters, expected_d_filters
        # ), f"Expected {expected_d_filters}, but got {conv.d_filters}"
        # assert np.allclose(
        #     conv.d_biases, expected_d_biases
        # ), f"Expected {expected_d_biases}, but got {conv.d_biases}"
