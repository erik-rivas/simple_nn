import numpy as np

from neural_network.loss_functions.loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss.
    """

    epsilon: float = 1e-2

    def calculate(self, y_pred, y_true):
        """
        Calculate categorical cross-entropy loss for all the samples.
        """
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)

        return data_loss

    def forward(self, y_pred, y_true):
        """
        # https://gombru.github.io/2018/05/23/cross_entropy_loss/
        y_pred is output from the softmax layer
        y_true is one-hot encoded
        """

        # Clipping predictions to prevent log of zero
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Compute loss
        loss = -np.sum(y_true * np.log(y_pred_clipped))
        loss /= y_true.shape[0]

        return loss

    def backward(self, y_pred, y_true):
        """
        y_pred is output from the softmax layer
        y_true is one-hot encoded
        """
        samples = len(y_pred)
        labels = len(y_pred[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / y_pred
        self.dinputs = self.dinputs / samples

        return self.dinputs

    def __str__(self) -> str:
        return f"Loss CategoricalCrossentropy"
