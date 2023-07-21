import numpy as np

from neural_network.layers.layer_base import Layer


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        """
        Initializes a Conv2D layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (i.e., number of filters).
            kernel_size (int): Size of the square filter.
            stride (int, optional): Stride size. Defaults to 1.
            padding (int, optional): Padding size. Defaults to 0.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filters = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.biases = np.random.randn(out_channels, 1)

    def set_weights_biases(self, filters, biases):
        self.filters = filters
        self.biases = biases

    def forward(self, input_data):
        """
        Performs a forward pass through the Conv2D layer.

        Args:
            input (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output data.
        """
        self.last_input_data = input_data

        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, filter_height, filter_width = self.filters.shape

        out_height = (
            int((in_height - filter_height + 2 * self.padding) / self.stride) + 1
        )
        out_width = int((in_width - filter_width + 2 * self.padding) / self.stride) + 1

        self.last_input_data_padded = np.pad(
            input_data,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )

        output_data = np.zeros((batch_size, out_channels, out_height, out_width))

        for c_out in range(out_channels):
            for c_in in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_window = self.last_input_data_padded[
                            :,
                            c_in,
                            i * self.stride : i * self.stride + filter_height,
                            j * self.stride : j * self.stride + filter_width,
                        ]
                        filters = self.filters[c_out, c_in, :, :]
                        gradients = input_window * filters + self.biases[c_out]
                        output_data[:, c_out, i, j] += np.sum(gradients, axis=(1, 2))

        return output_data

    def backward(self, d_out):
        """
        Perform a backward pass through the Conv2D layer.

        The backward pass computes the gradient of the loss function with respect to the input,
        filters, and biases of the Conv2D layer. It uses these gradients to update the filters
        and biases.

        Args:
            dL_dy (numpy.ndarray): The derivative of the loss function with respect to the output of the Conv2D layer.

        Returns:
            d_input (numpy.ndarray): The derivative of the convolution with respect to the input of the Conv2D layer.
        """
        # Get the shapes of the filters and the derivative of the input

        batch_size, out_channels, out_height, out_width = d_out.shape
        _, in_channels, filter_height, filter_width = self.filters.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(self.last_input_data_padded, dtype=float)
        # Loop over the output channels, height, and width
        # To calculate the derivative of the layer with respect to the filters we need to
        # - calculate the convolution between the input and the derivative of the output
        # - using the filter weight and height as windows

        for c_out in range(out_channels):
            for c_in in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_window = self.last_input_data_padded[
                            :,
                            c_in,
                            i * self.stride : i * self.stride + filter_height,
                            j * self.stride : j * self.stride + filter_width,
                        ]
                        gradients = input_window * d_out[:, c_out, i, j, None, None]
                        d_filters[c_out, c_in, :, :] += np.sum(
                            gradients, axis=(0, 1, 2)
                        )
                        d_biases[c_out] += np.sum(d_out[:, c_out, i, j])

                        # Calculate the gradient for the input data
                        filters = self.filters[c_out, c_in, :, :]
                        d_out_window = (
                            d_out[:, c_out, i, j]
                            .reshape(-1, 1, 1)
                            .repeat(self.kernel_size, axis=1)
                            .repeat(self.kernel_size, axis=2)
                        )
                        # print(f"filters.shap:e{filters.shape}, d_out_window.shape:{d_out_window.shape}")
                        gradients = filters * d_out_window

                        d_input_padded[
                            :,
                            c_in,
                            i * self.stride : i * self.stride + filter_height,
                            j * self.stride : j * self.stride + filter_width,
                        ] += np.sum(gradients)

        d_input = d_input_padded[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]

        self.d_filters = d_filters
        self.d_biases = d_biases

        return d_filters, d_biases, d_input

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
        self.biases -= learning_rate * self.d_biases
