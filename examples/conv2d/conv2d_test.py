import matplotlib.pyplot as plt
import numpy as np

from examples.conv2d.conv2d_train import test_filter_noise_training
from neural_network.layers.layer_conv2d import Conv2D


def test1():
    out_channels = 1
    in_channels = 1
    kernel_size = 3

    # Initialize a Conv2D layer
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    conv.filters = np.zeros(
        (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
    )
    conv.filters[0, 0, 1, :] = 1

    conv.biases = np.zeros(
        (
            out_channels,
            1,
        )
    )

    # Define a simple 5x5 input
    input = np.zeros(
        (
            1,
            1,
            10,
            10,
        )
    )
    input[0, 0, 2, :] = 0.9
    print("Input:\n", input[0])

    # plt.imshow(input[0, 0], cmap="gray")
    # plt.title("Input")
    # plt.show()

    # plt.imshow(conv.filters[0], cmap="gray")
    # plt.title("Filter")
    # plt.show()

    # Perform a forward pass
    output = conv.forward(input)

    # # Print and plot the output
    # print("Output:\n", output[0])
    # plt.imshow(output[0, 0], cmap="gray")
    # plt.title("Output of Forward Pass")
    # plt.show()

    # Define a simple upstream gradient that matches the shape of the output
    dL_dy = np.ones_like(output) * 0
    dL_dy[0, 0, 2, :] = 1

    # Perform a backward pass
    d_input = conv.backward(dL_dy)

    # Print and plot the input gradient
    print("Input Gradient:\n", d_input[0, 0])
    plt.imshow(d_input[0, 0], cmap="gray")
    plt.title("Gradient of Input from Backward Pass")
    plt.show()

    # Perform an update
    conv.update(learning_rate=0.01)

    # Print the updated filters and biases
    print("Updated Filters:\n", conv.filters)
    print("Updated Biases:\n", conv.biases)


def test2():
    # Unit test

    # Create a 5x5 image with 1 channel
    # Form a square figure of ones in a matrix of 5x5 with 0's and 1's
    input_data = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    input_data = input_data[np.newaxis, :]

    # Create a Conv2D layer with 1 input channel, 2 output channels, a 3x3 kernel, stride of 1, and padding of 1
    # Initialize the kernels for a vertical line and horizontal line detectors
    conv2d_layer = Conv2D(1, 2, 3, stride=1, padding=1)

    conv2d_layer.filters[0, 0, :, :] = np.array(
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    )  # vertical line detector

    conv2d_layer.filters[1, 0, :, :] = np.array(
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    )  # horizontal line detector

    # Forward pass
    output_data = conv2d_layer.forward(input_data)

    # Plot the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(input_data[0, 0, :, :], cmap="gray")
    axs[0].set_title("Original Image")

    axs[1].imshow(output_data[0, 0, :, :], cmap="gray")
    axs[1].set_title("Vertical Line Detection")

    axs[2].imshow(output_data[0, 1, :, :], cmap="gray")
    axs[2].set_title("Horizontal Line Detection")

    for ax in axs:
        ax.axis("off")

    plt.show()

    # Check output shape
    assert output_data.shape == (1, 2, 5, 5), "Output shape is incorrect"

    # Backward pass
    d_out = np.ones((1, 2, 5, 5))  # use ones instead of random numbers for simplicity
    d_filters, d_biases, d_input = conv2d_layer.backward(d_out)

    # Check gradient shapes
    assert d_filters.shape == conv2d_layer.filters.shape, "d_filters shape is incorrect"
    assert d_biases.shape == conv2d_layer.biases.shape, "d_biases shape is incorrect"
    assert d_input.shape == input_data.shape, "d_input shape is incorrect"

    # Return the outputs
    return output_data, d_filters, d_biases, d_input


def test_line_detector():
    # Create a 7x7 image with 1 channel
    # Form a cross figure of ones in a matrix of 7x7 with 0's and 1's
    input_data = np.array(
        [
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ]
    )

    input_data = input_data[np.newaxis, :]

    # Create a Conv2D layer with 1 input channel, 2 output channels, a 3x3 kernel, stride of 1, and padding of 1
    conv2d_layer = Conv2D(1, 2, 3, stride=1, padding=1)

    # Initialize the kernels for a vertical line and horizontal line detectors
    conv2d_layer.filters[0, 0, :, :] = np.array(
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    )  # vertical line detector

    conv2d_layer.filters[1, 0, :, :] = np.array(
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    )  # horizontal line detector

    # Forward pass
    output_data = conv2d_layer.forward(input_data)

    # Check output shape
    assert output_data.shape == (1, 2, 7, 7), "Output shape is incorrect"

    # Backward pass
    d_out = np.ones((1, 2, 7, 7)) + np.random.normal(scale=0.1, size=output_data.shape)

    d_filters, d_biases, d_input = conv2d_layer.backward(d_out)

    # Check gradient shapes
    # assert d_filters.shape == conv2d_layer.filters.shape, "d_filters shape is incorrect"
    # assert d_biases.shape == conv2d_layer.biases.shape, "d_biases shape is incorrect"
    # assert d_input.shape == input_data.shape, "d_input shape is incorrect"

    # Plot the original image
    plt.figure(figsize=(5, 5))
    plt.imshow(input_data[0, 0, :, :], cmap="gray")
    plt.title("Original Image")
    plt.show()

    # Plot the output images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(output_data[0, 0, :, :], cmap="gray")
    axs[0].set_title("Vertical Line Detection")
    axs[1].imshow(output_data[0, 1, :, :], cmap="gray")
    axs[1].set_title("Horizontal Line Detection")
    plt.show()

    # Return the outputs
    return output_data, d_filters, d_biases, d_input


def test4():
    # Run the new test
    output_data, d_filters, d_biases, d_input = test_line_detector()
    output_data, d_filters, d_biases, d_input


def run():
    # test1()
    # test2()
    test4()
    test_filter_noise_training()
