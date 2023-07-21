import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.filters = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.biases = np.random.randn(out_channels, 1)

        self.stride = stride
        self.padding = padding

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
                        d_input_padded[
                            :,
                            c_in,
                            i * self.stride : i * self.stride + filter_height,
                            j * self.stride : j * self.stride + filter_width,
                        ] += np.sum(
                            self.filters[c_out, c_in, :, :, None, None]
                            * d_out[:, c_out, i, j, None, None, None],
                            axis=(1, 2),
                        )

        d_input = d_input_padded[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]

        return d_filters, d_biases, d_input

    def update(self, d_filters, d_biases, learning_rate):
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
