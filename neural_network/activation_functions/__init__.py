from neural_network.activation_functions.activation import ActivationFunction
from neural_network.activation_functions.relu import Activation_ReLU
from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.activation_functions.softmax import Activation_SoftMax

__all__ = [
    "ActivationFunction",
    "Activation_ReLU",
    "Activation_Sigmoid",
    "Activation_SoftMax",
]

activation_fn_map = {
    "relu": Activation_ReLU,
    "sigmoid": Activation_Sigmoid,
    "softmax": Activation_SoftMax,
}
