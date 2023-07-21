import matplotlib.pyplot as plt
import numpy as np

from neural_network.layers.max_pool import MaxPool2D


def run():
    # Initialize a MaxPooling2D layer
    pool = MaxPool2D(pool_size=2, stride=2)

    # Define a simple 4x4 input
    input = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]])
    print("Input:\n", input[0])

    # Perform a forward pass
    output = pool.forward(input)

    # Print and plot the output
    print("Output:\n", output[0])
    plt.imshow(output[0], cmap="gray")
    plt.title("Output of Forward Pass")
    plt.show()

    # Define a simple upstream gradient that matches the shape of the output
    dL_dy = np.ones_like(output)

    # Perform a backward pass
    d_input = pool.backward(dL_dy)

    # Print and plot the input gradient
    print("Input Gradient:\n", d_input[0])
    plt.imshow(d_input[0], cmap="gray")
    plt.title("Gradient of Input from Backward Pass")
    plt.show()
