import numpy as np


class Layer_Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
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
        batch_size, out_channels, out_height, out_width = d_out.shape
        _, in_channels, filter_height, filter_width = self.filters.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(self.last_input_data_padded, dtype=float)

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
        self.filters -= learning_rate * self.d_filters
        self.biases -= learning_rate * self.d_biases
