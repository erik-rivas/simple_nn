from neural_network.accuracy.accuracy import Accuracy
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError
from neural_network.optimizers.optimizer_sgd import Optimizer_SGD


class NeuralNetwork:
    layers = None

    def __init__(self, layers, loss_fn=None, learning_rate=0.1):
        self.layers = layers
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        # Set default optimizers and loss_fn
        self.optimizer = Optimizer_SGD(learning_rate)
        if not loss_fn:
            self.loss_fn = Loss_MeanSquaredError()
        self.accuracy = Accuracy()

    def forward(self, X):
        for layer in self.layers:
            print(f"Forwarding layer: {layer}")
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

    def train(self, X, y_true, epochs=1000, print_every=100):
        dense = self.layers[0]
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn.calculate(y_pred, y_true)
            # accuracy = self.accuracy.calculate(y_pred, y)
            accuracy = 0.001

            self.backward(y_pred, y_true)
            self.update()

            if epoch % print_every == 0:
                print(f"epoch: {epoch}, loss: {loss}")
