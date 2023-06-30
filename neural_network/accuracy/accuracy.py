import numpy as np


class Accuracy:
    def __call__(self, predictions, targets):
        # round predictions to the closest integer (0 or 1 for binary classification)
        predictions = np.round(predictions)
        return np.mean(predictions == targets)

    def calculate(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == y_true)

        return accuracy
