from enum import Enum

from neural_network.activation_functions.activation import ActivationFunction
from neural_network.activation_functions.relu import Activation_ReLU
from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.activation_functions.softmax import Activation_SoftMax
from neural_network.activation_functions.tanh import Activation_Tanh


class ActivationFunctions(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    TANH = "tanh"


activation_fn_map = {
    ActivationFunctions.RELU: Activation_ReLU,
    ActivationFunctions.SIGMOID: Activation_Sigmoid,
    ActivationFunctions.SOFTMAX: Activation_SoftMax,
    ActivationFunctions.TANH: Activation_Tanh,
}

temp_dict = {str(key): act_fn for key, act_fn in activation_fn_map.items()}

activation_fn_map.update(temp_dict)

__all__ = [
    "ActivationFunction",
    "Activation_ReLU",
    "Activation_Sigmoid",
    "Activation_SoftMax",
    "Activation_Tanh",
]
