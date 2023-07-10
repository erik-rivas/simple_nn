import numpy as np
from numpy.typing import NDArray

from neural_network.accuracy.accuracy import Accuracy
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError


class NeuralNetwork:
    layers = None

    def __init__(self, layers, loss_fn=None):
        self.layers = layers
        self.loss_fn = loss_fn

        # Set default optimizers and loss_fn
        if not loss_fn:
            self.loss_fn = Loss_MeanSquaredError()
        self.accuracy = Accuracy()

    def forward(self, X) -> NDArray:
        for layer in self.layers:
            # print(layer)
            X = layer.forward(X)

        return layer.output

    def backward(self, y_pred, y_true):
        loss_derivative = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            loss_derivative = layer.backward(loss_derivative)
        return loss_derivative

    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def train(self, X, y_true, learning_rate=0.001, epochs=1000, print_every=100):
        self.learning_rate = learning_rate

        losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn.calculate(y_pred, y_true)
            losses.append(loss)
            # accuracy = self.accuracy.calculate(y_pred, y)
            # accuracy = 0.001

            self.backward(y_pred, y_true)
            self.update()

            if epoch % print_every == 0:
                print(f"epoch: {epoch}, loss: {loss}")

        return np.mean(losses)
