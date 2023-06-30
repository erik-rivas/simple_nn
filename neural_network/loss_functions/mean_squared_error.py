import numpy as np


class Loss_MeanSquaredError:
    def calculate(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size
