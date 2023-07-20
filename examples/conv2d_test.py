import matplotlib.pyplot as plt
import numpy as np

from neural_network.layers.layer_conv2d import Layer_Conv2D


def run():
    out_channels = 1
    in_channels = 1
    kernel_size = 3

    # Initialize a Conv2D layer
    conv = Layer_Conv2D(
        in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
    )

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
            10,
            10,
        )
    )
    input[0, 2, :] = 0.9
    print("Input:\n", input[0])
    plt.imshow(input[0], cmap="gray")
    plt.title("Input")
    plt.show()

    plt.imshow(conv.filters[0], cmap="gray")
    plt.title("Filter")
    plt.show()

    # Perform a forward pass
    output = conv.forward(input)

    # Print and plot the output
    print("Output:\n", output[0])
    plt.imshow(output[0], cmap="gray")
    plt.title("Output of Forward Pass")
    plt.show()

    # Define a simple upstream gradient that matches the shape of the output
    dL_dy = np.ones_like(output) * 0
    dL_dy[0, 2, :] = 1

    # Perform a backward pass
    d_input = conv.backward(dL_dy)

    # Print and plot the input gradient
    print("Input Gradient:\n", d_input[0])
    plt.imshow(d_input[0], cmap="gray")
    plt.title("Gradient of Input from Backward Pass")
    plt.show()

    # Perform an update
    conv.update(learning_rate=0.01)

    # Print the updated filters and biases
    print("Updated Filters:\n", conv.filters)
    print("Updated Biases:\n", conv.biases)
