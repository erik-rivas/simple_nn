import numpy as np

from neural_network.loss_functions.loss_function import LossFunction


class Loss_MeanSquaredError(LossFunction):
    def calculate(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size
