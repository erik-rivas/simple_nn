import numpy as np

from neural_network.layers.max_pool import MaxPool2D


def test_maxpool2d():
    # Create a MaxPool2D layer with a 2x2 pooling window and stride of 2
    maxpool2d = MaxPool2D(pool_size=2, stride=2)

    # Create a simple 4x4 input matrix
    input_data = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )

    # Compute the output of the MaxPool2D layer
    output_data = maxpool2d.forward(input_data)

    # Expected output is the maximum of each 2x2 block of the input
    expected_output = np.array([[[[6, 8], [14, 16]]]])

    # Assert that the output matches the expected output
    assert np.allclose(
        output_data, expected_output
    ), f"Expected {expected_output}, but got {output_data}"

    # Compute the gradient of the loss with respect to the input
    d_input = maxpool2d.backward(expected_output)

    # Expected gradient is a 4x4 matrix with the gradients distributed according to the locations of the maxima
    expected_d_input = np.array(
        [[[[0, 0, 0, 0], [0, 6, 0, 8], [0, 0, 0, 0], [0, 14, 0, 16]]]]
    )

    # Assert that the computed gradient matches the expected gradient
    assert np.allclose(
        d_input, expected_d_input
    ), f"Expected {expected_d_input}, but got {d_input}"


def run():
    test_maxpool2d()
