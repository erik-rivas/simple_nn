import numpy as np

from neural_network.layers.layer_conv2d import Layer_Conv2D


class TestLayerConv2d:
    # def test_forward(self):
    #     layer = Layer_Conv2D(in_channels=1, out_channels=1, kernel_size=2, padding=1)
    #     input = np.array(
    #         [
    #             [
    #                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    #             ]
    #         ]
    #     )
    #     layer.kernel = np.ones((1, 1, 2, 2)) * 0.1

    #     output = layer.forward(input)
    #     expected_output = np.array(
    #         [
    #             [
    #                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    #             ]
    #         ]
    #     )
    #     assert np.allclose(output, expected_output, atol=1e-6)

    # def test_backward(self):
    #     layer = Layer_Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    #     input = np.array(
    #         [
    #             [
    #                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    #             ]
    #         ]
    #     )

    def test_forward_backward(self):
        # Initialize Conv2D layer
        conv = Layer_Conv2D(
            in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1
        )

        # Set filters and biases to known values
        conv.filters = np.array([[[[1, 0], [0, 1]]]])
        conv.biases = np.array([[1]])

        # Forward pass
        input = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        output = conv.forward(input)

        # Expected output
        expected_output = np.array([[[2, 3, 4], [5, 7, 9], [8, 13, 15]]])
        assert np.allclose(
            output, expected_output
        ), f"Expected {expected_output}, but got {output}"

        # Backward pass
        dL_dy = np.full((1, 3, 3), 0.1)
        d_input = conv.backward(dL_dy)

        # Expected gradients
        expected_d_filters = np.array([[[[3.1, 4.6], [6.1, 7.6]]]])
        expected_d_biases = np.array([[0.9]])
        expected_d_input = np.full((1, 3, 3), 0.2)

        assert np.allclose(
            conv.d_filters, expected_d_filters
        ), f"Expected {expected_d_filters}, but got {conv.d_filters}"
        assert np.allclose(
            conv.d_biases, expected_d_biases
        ), f"Expected {expected_d_biases}, but got {conv.d_biases}"
        assert np.allclose(
            d_input, expected_d_input
        ), f"Expected {expected_d_input}, but got {d_input}"
