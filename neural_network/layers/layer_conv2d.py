import numpy as np

from neural_network.layers.layer_base import Layer


class Layer_Conv2D(Layer):
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
        n_samples, in_channels, h_i, w_i = input.shape
        out_channels, _, h_f, w_f = self.filters.shape

        # Add padding to input height and width
        h_o = (h_i - h_f + 2 * self.padding) // self.stride + 1
        w_o = (w_i - w_f + 2 * self.padding) // self.stride + 1

        self.output = np.zeros((n_samples, out_channels, h_o, w_o))

        self.padded_input = np.pad(
            input,
            pad_width=(
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
            constant_values=0,
        )

        for c_out in range(out_channels):
            for i in range(0, h_o * self.stride, self.stride):
                for j in range(0, w_o * self.stride, self.stride):
                    pixel_ouput = np.zeros(n_samples)
                    for c_in in range(in_channels):
                        curr_batch_imgs = self.padded_input[:, c_in]
                        # Get the slices
                        img_slice = curr_batch_imgs[:, i : i + h_f, j : j + w_f]
                        current_filter = self.filters[c_out, c_in]
                        # Calculate the product of the filter and the slice
                        convolution = img_slice * current_filter
                        # Sum the values of the convolution
                        convolution_sum = np.sum(convolution, axis=(1, 2))
                        # convolution_sum = convolution_sum.reshape(-1, 1)
                        # Add the bias
                        convolution_sum += self.biases[c_out]
                        # Apply the activation function

                        pixel_ouput += convolution_sum

                    self.output[
                        :, c_out, i // self.stride, j // self.stride
                    ] = pixel_ouput

                    # output_view[i // self.stride, j // self.stride] = pixel_ouput

            # self.output[c_out, :, :] += self.biases[c_out]

        return self.output

    def _convolution(self, input_section, filter):
        return np.sum(input_section * filter)

    def backward(self, dL_dy):
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
        out_channels, _, h_f, w_f = self.filters.shape
        n_samples, in_channels, h_i, w_i = self.input.shape

        # Initialize arrays to hold the gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)
        self.d_input = np.zeros_like(self.input)

        # Pad the derivative of the input to handle the edges of the input during the convolution
        d_input = np.zeros_like(self.input)
        padded_d_input = np.pad(
            self.d_input,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
        )

        # Loop over the output channels, height, and width
        # To calculate the derivative of the layer with respect to the filters we need to
        # - calculate the convolution between the input and the derivative of the output
        # - using the filter weight and height as windows

        for c_out in range(out_channels):
            for i in range(0, h_i * self.stride, self.stride):
                for j in range(0, w_i * self.stride, self.stride):
                    for c_in in range(in_channels):
                        self.d_filters[c_out, c_in] += (
                            dL_dy[c_out, i // self.stride, j // self.stride]
                            * self.padded_input[c_in, i : i + h_f, j : j + w_f]
                        )

            self.d_biases[c_out] = np.sum(dL_dy[c_out, :, :])

            for i in range(h_f):
                for j in range(w_f):
                    for c_in in range(in_channels):
                        padded_d_input[
                            c_in,
                            i : i + h_i * self.stride : self.stride,
                            j : j + w_i * self.stride : self.stride,
                        ] += dL_dy[c_out, i // self.stride, j // self.stride] * np.flip(
                            self.filters[c_out, c_in, i, j]
                        )

        self.d_input = padded_d_input[
            :, self.padding : h_i + self.padding, self.padding : w_i + self.padding
        ]

        return self.d_input

        self.d_input = d_input

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
