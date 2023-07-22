from neural_network.layers.layer_base import Layer


class Layer_Reshape(Layer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        self.last_input_data = inputs

        output = inputs.reshape(self.shape)
        self.last_output_data = output

        return output

    def backward(self, dL_dy):
        return dL_dy.reshape(self.inputs_shape)

    def update(self, _: float):
        pass

    def __str__(self) -> str:
        return f"||Reshape: {self.last_input_data.shape} => {self.last_output_data.shape} ||"
