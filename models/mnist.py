from matplotlib import pyplot as plt

from libs.idx import read_idx
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from neural_network.neural_network import NeuralNetwork


class MnistModel(NeuralNetwork):
    def __init__(self, items_to_read=10):
        self.n_features = 784
        self.n_classes = 10

        self.layers = [
            Layer_Dense(n_features=self.n_features, n_neurons=64, activation_fn="relu"),
            Layer_Dense(n_features=64, n_neurons=10, activation_fn="softmax"),
        ]
        self.loss_fn = CategoricalCrossEntropy()

        super().__init__(self.layers, loss_fn=self.loss_fn)

        self.get_mnist(items_to_read)

    def get_mnist(self, items_to_read=10):
        self.X_train, _ = read_idx(
            path="data/mnist/train-images-idx3-ubyte", items_to_read=items_to_read
        )

        self.y_train, _ = read_idx(
            "data/mnist/train-labels-idx1-ubyte", items_to_read=items_to_read
        )

        return self.X_train, self.y_train

    def plot_dataset_sample(self, index):
        plt.imshow(self.X_train[index].reshape((28, 28)), cmap="gray")
        plt.title(self.y_train[index])
        plt.show()

    def train(self, epochs=1000, batch_size=128, learning_rate=0.01, verbose=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.history = {
            "loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i : i + batch_size]
                batch_y = self.y_train[i : i + batch_size]

                y_pred = self.forward(batch_X)
                loss = self.loss_fn.calculate(y_pred, batch_y)
                accuracy = self.accuracy.calculate(y_pred, batch_y)

                self.backward(y_pred, batch_y)
                self.update()

            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            if verbose and epoch % verbose == 0:
                print(
                    f"epoch: {epoch}, "
                    + f"loss: {loss:.3f}, "
                    + f"accuracy: {accuracy:.3f}"
                )
