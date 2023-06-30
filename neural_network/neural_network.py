# from neural_network.activation_functions import Activation_Softmax, Activation_ReLU
# from neural_network.layers import (
#     Loss_CategoricalCrossentropy_Loss,
#     Layer_Dense,
# )
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
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn.calculate(y_pred, y_true)
            # accuracy = self.accuracy.calculate(y_pred, y)
            accuracy = 0.001

            self.backward(y_pred, y_true)
            self.update()

            if epoch % print_every == 0:
                print(
                    f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}"
                )


# class NeuralNetwork:
#     def __init__(self, layers, loss, seed=1):
#         self.layers = layers
#         self.loss = loss
#         self.seed = seed
#         self.optimizer = Optimizer_SGD()

#     def predict(self, X):
#         return self.forward(X, training=False)

#     def forward(self, X, training=True):
#         layer_outputs = []
#         current_output = X
#         for layer in self.layers:
#             current_output = layer.forward(current_output, training)
#             layer_outputs.append(current_output)
#         return layer_outputs[-1]

#     def backward(self, y_pred, y_true):
#         loss_derivative = self.loss.backward(y_pred, y_true)
#         for layer in reversed(self.layers):
#             loss_derivative = layer.backward(loss_derivative)
#         return loss_derivative

#     def update(self):
#         self.optimizer.update_params(self.layers)

#     def train(self, X, y, epochs, print_every=100):
#         np.random.seed(self.seed)
#         for epoch in range(epochs):
#             y_pred = self.predict(X)
#             loss = self.loss.loss(y_pred, y)
#             self.backward(y_pred, y)
#             self.update()
#             if epoch % print_every == 0:
#                 print('Epoch: {}, Loss: {}'.format(epoch, loss))
