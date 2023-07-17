from enum import Enum

from neural_network.activation_functions.activation import ActivationFunction
from neural_network.activation_functions.relu import Activation_ReLU
from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.activation_functions.softmax import Activation_SoftMax


class ActivationFunctions(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


activation_fn_map = {
    ActivationFunctions.RELU: Activation_ReLU,
    ActivationFunctions.SIGMOID: Activation_Sigmoid,
    ActivationFunctions.SOFTMAX: Activation_SoftMax,
}

__all__ = [
    "ActivationFunction",
    "Activation_ReLU",
    "Activation_Sigmoid",
    "Activation_SoftMax",
]
