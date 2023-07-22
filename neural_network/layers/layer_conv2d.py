import numpy as np

from libs.img_proc import im2col_indices
from neural_network.layers.layer_base import Layer


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        """
        Initialize a 2D convolutional layer.

        Parameters:
        - in_channels: number of input channels.
        - out_channels: number of output channels (also number of filters).
        - kernel_size: size of filters (considered to be square).
        - stride: the stride of the convolution. Default is 1.
        - padding: zero-padding added on both sides of the input. Default is 1.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize filters and biases with random values
        self.filters = np.random.normal(
            scale=0.1,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.biases = np.random.normal(
            scale=0.1,
            size=(out_channels, 1),
        ).astype(np.float32)

    def set_weights_biases(self, filters, biases):
        """
        Set filters and biases for this layer manually.

        Parameters:
        - filters: 4D array of shape (out_channels, in_channels, kernel_height, kernel_width)
        - biases: 2D array of shape (out_channels, 1)
        """
        self.filters = filters.astype(np.float32)
        self.biases = biases.astype(np.float32)

    def forward(self, input_data):
        """
        Perform a forward pass of the conv layer using the given input.

        Parameters:
        - input_data: a numpy array with shape (batch_size, in_channels, input_height, input_width)
        Returns:
        - a numpy array with shape (batch_size, out_channels, output_height, output_width)
        """
        self.last_input_data = input_data

        # Get dimensions from input data and filter
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, filter_height, filter_width = self.filters.shape

        # Calculate output dimensions
        out_height = (
            int((in_height - filter_height + 2 * self.padding) / self.stride) + 1
        )
        out_width = int((in_width - filter_width + 2 * self.padding) / self.stride) + 1

        # Convert the input data and filters to column format
        self.last_input_data_col = im2col_indices(
            input_data,
            filter_height,
            filter_width,
            padding=self.padding,
            stride=self.stride,
        )
        self.filters_col = self.filters.reshape(out_channels, -1)

        # Compute the output data in column format
        output_data_col = self.filters_col @ self.last_input_data_col + self.biases

        # Reshape the output data to the output dimensions
        output_data = output_data_col.reshape(
            out_channels, out_height, out_width, batch_size
        )
        output_data = output_data.transpose(3, 0, 1, 2)

        # Save the shape of the output data for the backward pass
        self.last_output_data_shape = output_data.shape

        return output_data

    def backward(self, d_out):
        """
        Perform a backward pass of the conv layer.

        Parameters:
        - d_out: the loss gradient for this layer's outputs, with shape (batch_size, out_channels, out_height, out_width)
        Returns:
        - loss gradients for this layer's filters, biases, and input data
        """
        batch_size, out_channels, _, _ = d_out.shape

        # Compute the loss gradient for the biases
        db = np.sum(d_out, axis=(0, 2, 3))

        # Reshape the loss gradient to a 2D array
        d_out_col = d_out.transpose(1, 2, 3, 0).reshape(out_channels, -1)

        # Compute the loss gradients for the filters
        dw = d_out_col @ self.last_input_data_col.T
        dw = dw.reshape(self.filters.shape)

        # Compute the loss gradient for the input data
        dx_col = self.filters_col.T @ d_out_col
        dx_col = dx_col.reshape(
            batch_size,
            self.kernel_size,
            self.kernel_size,
            self.in_channels,
            self.last_output_data_shape[2],
            self.last_output_data_shape[3],
        )
        dx = np.zeros(
            (
                batch_size,
                self.in_channels,
                self.last_output_data_shape[2] + 2 * self.padding,
                self.last_output_data_shape[3] + 2 * self.padding,
            )
        )

        for i in range(self.kernel_size):
            i_max = i + self.stride * self.last_output_data_shape[2]
            for j in range(self.kernel_size):
                j_max = j + self.stride * self.last_output_data_shape[3]
                dx[:, :, i : i_max : self.stride, j : j_max : self.stride] += dx_col[
                    :, i, j, :, :, :
                ]

        dx = dx[
            :,
            :,
            self.padding : self.padding + self.last_output_data_shape[2],
            self.padding : self.padding + self.last_output_data_shape[3],
        ]

        # Save the computed gradients
        self.d_filters = dw
        self.d_biases = db.reshape(-1, 1)

        return dx

    def update(self, learning_rate):
        """
        Update the filters and biases using gradient descent.

        Parameters:
        - learning_rate: learning rate for gradient descent.
        """
        self.filters -= learning_rate * self.d_filters
        self.biases -= learning_rate * self.d_biases

    def __str__(self) -> str:
        return f"||Conv2D: {self.last_input_data.shape} => {self.last_output_data_shape} ||"
