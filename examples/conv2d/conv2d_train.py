# Define the mean squared error loss and its derivative
import numpy as np
from matplotlib import pyplot as plt

from neural_network.layers.layer_conv2d import Conv2D


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def train_conv2d(input_data, n_iterations=5000, learning_rate=0.1):
    # Create a Conv2D layer with 1 input channel, 2 output channels, a 3x3 kernel, stride of 1, and padding of 1
    conv2d_layer = Conv2D(1, 2, 3, stride=1, padding=1)

    # Initialize the kernels for a vertical line and horizontal line detectors
    conv2d_layer.filters[0, 0, :, :] = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )  # vertical line detector

    conv2d_layer.filters[1, 0, :, :] = np.array(
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    )  # horizontal line detector

    # Forward pass to get the "true" output
    y_true = conv2d_layer.forward(input_data)

    # Plot the "true" output
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(y_true[0, 0, :, :], cmap="gray")
    axs[0].set_title("Vertical Line Detection (True)")
    axs[1].imshow(y_true[0, 1, :, :], cmap="gray")
    axs[1].set_title("Horizontal Line Detection (True)")
    plt.show()

    # Add noise to the filters
    noise = np.random.normal(scale=10, size=conv2d_layer.filters.shape)
    conv2d_layer.filters = noise

    # Gradient descent loop
    for i in range(n_iterations):
        # Forward pass
        y_pred = conv2d_layer.forward(input_data)

        # Compute the loss
        loss = mse_loss(y_true, y_pred)

        # Backward pass
        d_out = mse_loss_derivative(y_true, y_pred)
        _ = conv2d_layer.backward(d_out)

        # Update the filters and biases using gradient descent
        conv2d_layer.update(learning_rate)

        # Print the loss every 500 iterations
        if i % 500 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    # Forward pass to get the final output
    y_pred = conv2d_layer.forward(input_data)

    # Plot the final output
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(y_pred[0, 0, :, :], cmap="gray")
    axs[0].set_title("Vertical Line Detection (Final)")
    axs[1].imshow(y_pred[0, 1, :, :], cmap="gray")
    axs[1].set_title("Horizontal Line Detection (Final)")
    plt.show()

    # Print the updated filters
    print("\nUpdated filters:")
    print(conv2d_layer.filters)

    # Return the updated filters and biases
    return conv2d_layer.filters, conv2d_layer.biases


def test_filter_noise_training():
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

    # Train the Conv2D layer
    filters, biases = train_conv2d(input_data)
