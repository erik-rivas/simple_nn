import numpy as np

from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError
from neural_network.neural_network import NeuralNetwork


class LinearRegressor(NeuralNetwork):
    def __init__(self):
        self.dense = Layer_Dense(n_features=1, n_neurons=1)
        layers = [self.dense]
        loss_fn = Loss_MeanSquaredError()

        super().__init__(layers, loss_fn)

    def train(self, x_data, y_data, n_baches, bach_size=8, epochs=128):
        for _ in range(n_baches):
            rand_ind = np.random.randint(len(x_data), size=bach_size)
            x_sample = x_data[rand_ind].reshape((-1, 1))
            y_sample = y_data[rand_ind].reshape(-1, 1)
            super().train(
                X=x_sample,
                y_true=y_sample,
                learning_rate=0.01,
                epochs=epochs,
                print_every=1000,
            )

        return self.dense.weights, self.dense.biases
