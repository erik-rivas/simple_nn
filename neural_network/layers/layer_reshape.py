from neural_network.layers.layer_base import Layer


class Layer_Reshape(Layer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(self.shape)

    def backward(self, dL_dy):
        return dL_dy.reshape(self.inputs_shape)

    def update(self, _: float):
        pass
