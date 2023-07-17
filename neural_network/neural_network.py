import math

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from neural_network.accuracy.accuracy import Accuracy
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError


class NeuralNetwork:
    layers = None

    def __init__(self, layers, loss_fn=None, debug_verbose=False):
        """Initialize the NeuralNetwork with layers, loss function and learning rate."""
        self.layers = layers
        self.loss_fn = loss_fn
        self.debug_verbose = debug_verbose

        # Set default optimizers and loss_fn
        if not loss_fn:
            self.loss_fn = Loss_MeanSquaredError()

        self.accuracy = Accuracy()

    def forward(self, X) -> NDArray:
        """Forward propagation through the network."""
        for layer in self.layers:
            X = layer.forward(X)
            if self.debug_verbose:
                print(layer)

        return layer.output

    def backward(self, y_pred, y_true):
        """Backward propagation through the network."""
        loss_derivative = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            loss_derivative = layer.backward(loss_derivative)
        return loss_derivative

    def update(self):
        """Update weights in the network."""
        for layer in self.layers:
            layer.update(self.learning_rate)

    def train(
        self,
        X,
        y_true,
        learning_rate=0.001,
        epochs=1000,
        batch_size=128,
        iter_per_batch=10,
        verbose=100,
    ):
        """Train the neural network with given input and target output."""
        self.X_train = X
        self.y_train = y_true if len(y_true.shape) > 1 else y_true.reshape(-1, 1)

        # Set learning rate, batch size and epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.history = {
            "loss": [],
            "accuracy": [],
        }

        it = 0
        for epoch in range(epochs):
            for i_batch in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i_batch : i_batch + batch_size]
                batch_y = self.y_train[i_batch : i_batch + batch_size]

                for iteration in range(iter_per_batch):
                    y_pred = self.forward(batch_X)
                    loss = self.loss_fn.calculate(y_pred, batch_y)
                    if loss is None or math.isnan(loss):
                        raise ValueError("Loss is None or NaN")

                    accuracy, precision, recall, f1_score = self.accuracy.calculate(
                        y_pred, batch_y
                    )

                    self.backward(y_pred, batch_y)
                    self.update()

                    self.history["loss"].append(loss)
                    self.history["accuracy"].append(accuracy)

                    if verbose and it % verbose == 0:
                        print(
                            f"epoch: {epoch}, batch: {i_batch}, it: {it}, "
                            + f"loss: {loss:.3f}, "
                            + f"accuracy: {accuracy:.3f}, "
                            + f"precision: {precision:.3f}, "
                            + f"recall: {recall:.3f}, "
                            + f"f1_score: {f1_score:.3f}"
                        )
                    it += 1

    def evaluate(self, X, y_true):
        """Evaluate the model with given input and target output."""
        y_pred = self.forward(X)
        loss = self.loss_fn.calculate(y_pred, y_true)
        accuracy, precision, recall, f1_score = self.accuracy.calculate(y_pred, y_true)

        return accuracy, precision, recall, f1_score

    def predict(self, X):
        """Predict the output with given input."""
        return self.forward(X)

    def plot_history(self, show=True):
        """Plot the history of loss and accuracy."""
        plt.plot(self.history["loss"])

        if show:
            plt.show()
