import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initializes a Conv2D layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (i.e., number of filters).
            kernel_size (int): Size of the square filter.
            stride (int, optional): Stride size. Defaults to 1.
            padding (int, optional): Padding size. Defaults to 0.
        """
        self.filters = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.biases = np.random.randn(out_channels, 1)

        self.stride = stride
        self.padding = padding

    def forward(self, input):
        """
        Performs a forward pass through the Conv2D layer.

        Args:
            input (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output data.
        """
        self.input = input
        in_channels, h_i, w_i = input.shape
        out_channels, _, h_f, w_f = self.filters.shape

        h_o = (h_i - h_f + 2 * self.padding) // self.stride + 1
        w_o = (w_i - w_f + 2 * self.padding) // self.stride + 1
        self.output = np.zeros((out_channels, h_o, w_o))

        padded_input = np.pad(
            input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
        )

        for c_out in range(out_channels):
            for i in range(0, h_o * self.stride, self.stride):
                for j in range(0, w_o * self.stride, self.stride):
                    for c_in in range(in_channels):
                        self.output[
                            c_out, i // self.stride, j // self.stride
                        ] += np.sum(
                            padded_input[c_in, i : i + h_f, j : j + w_f]
                            * self.filters[c_out, c_in]
                        )
            self.output[c_out, :, :] += self.biases[c_out]

        return self.output

    def backward(self, dL_dy):
        """
        Perform a backward pass through the Conv2D layer.

        The backward pass computes the gradient of the loss function with respect to the input,
        filters, and biases of the Conv2D layer. It uses these gradients to update the filters
        and biases.

        Args:
            dL_dy (numpy.ndarray): The derivative of the loss function with respect to the output of the Conv2D layer.

        Returns:
            d_input (numpy.ndarray): The derivative of the loss function with respect to the input of the Conv2D layer.
        """
        # Get the shapes of the filters and the derivative of the input
        out_channels, _, h_f, w_f = self.filters.shape
        in_channels, h_i, w_i = self.output.shape

        # Initialize arrays to hold the gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)
        self.d_input = np.zeros_like(self.input)

        # Pad the derivative of the input to handle the edges of the input during the convolution

        # Loop over the output channels, height, and width
        for c_out in range(out_channels):
            for i in range(h_i):
                for j in range(w_i):
                    # Check if the slice would go beyond the boundaries of the input
                    if i + h_f > h_i or j + w_f > w_i:
                        continue

                    # Loop over the input channels
                    for c_in in range(in_channels):
                        # Update the gradient of the filters
                        self.d_filters[c_out, c_in] += (
                            dL_dy[c_out, i, j]
                            * self.input[c_in, i : i + h_f, j : j + w_f]
                        )

                        # Update the gradient of the input
                        self.d_input[c_in, i : i + h_f, j : j + w_f] += dL_dy[
                            c_out, i, j
                        ] * np.flip(self.filters[c_out, c_in], axis=(0, 1))

            # Update the gradient of the biases
            self.d_biases[c_out] = np.sum(dL_dy[c_out, :, :])

        return self.d_input

    def update(self, learning_rate):
        """
        Update the filters and biases of the Conv2D layer using the gradients computed in the backward pass.

        The update step is a crucial part of the training process where the weights (filters and biases in this case)
        are updated in the opposite direction of the gradient. This is done in order to minimize the loss function.

        Args:
            learning_rate (float): The learning rate parameter controls the size of the update steps.
        """
        # Update the filters using the gradient of the filters and the learning rate
        self.filters -= learning_rate * self.d_filters

        # Update the biases using the gradient of the biases and the learning rate
        self.biases -= learning_rate * self.d_biases
