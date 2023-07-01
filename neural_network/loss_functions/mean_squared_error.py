import numpy as np


class Loss_MeanSquaredError:
    def calculate(self, y_pred, y_true):
        return np.mean((y_pred.flatten() - y_true.flatten()) ** 2)

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred.flatten() - y_true.flatten()) / y_pred.size
