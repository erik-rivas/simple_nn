import numpy as np


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        """
        Initializes a MaxPooling2D layer.

        Args:
            pool_size (int, optional): The size of the pooling window. Defaults to 2.
            stride (int, optional): The stride size. Defaults to 2.
        """
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        """
        Performs a forward pass through the MaxPooling2D layer.

        Args:
            input (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output data.
        """
        self.input = input
        n_samples, in_channels, h_i, w_i = input.shape

        h_o = (h_i - self.pool_size) // self.stride + 1
        w_o = (w_i - self.pool_size) // self.stride + 1
        output = np.zeros((n_samples, in_channels, h_o, w_o))

        for i in range(0, h_i, self.stride):
            for j in range(0, w_i, self.stride):
                input_window = input[
                    :, :, i : i + self.pool_size, j : j + self.pool_size
                ]
                max_value = np.max(input_window, axis=(2, 3))
                output[:, :, i // self.stride, j // self.stride] = max_value

        return output

    def backward(self, dL_dy):
        """
        Perform a backward pass through the MaxPooling2D layer.

        Args:
            dL_dy (numpy.ndarray): The derivative of the loss function with respect to the output of the MaxPooling2D layer.

        Returns:
            d_input (numpy.ndarray): The derivative of the loss function with respect to the input of the MaxPooling2D layer.
        """
        n_samples, in_channels, h_i, w_i = self.input.shape

        d_input = np.zeros_like(self.input)

        for i in range(0, h_i, self.stride):
            for j in range(0, w_i, self.stride):
                window = self.input[
                    :, :, i : i + self.pool_size, j : j + self.pool_size
                ]
                max_val = np.max(window, axis=(2, 3))
                mask = np.equal(window, max_val[:, :, None, None])

                gradient = (
                    mask * dL_dy[:, :, i // self.stride, j // self.stride, None, None]
                )

                d_input[:, :, i : i + self.pool_size, j : j + self.pool_size] = gradient

        return d_input

    def update(self, _):
        """
        This layer has no weights to update.
        """
        pass
