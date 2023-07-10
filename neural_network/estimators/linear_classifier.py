import numpy as np

from neural_network.activation_functions.sigmoid import Activation_Sigmoid
from neural_network.layers.layer_dense import Layer_Dense
from neural_network.loss_functions.mean_squared_error import Loss_MeanSquaredError
from neural_network.neural_network import NeuralNetwork


class LinearClassifier(NeuralNetwork):
    def __init__(self, n_features, random_state=None):
        self.n_features = n_features
        self.layer_dense = Layer_Dense(
            n_features=n_features,
            n_neurons=1,
            random_state=random_state,
        )
        # self.layer_dense.set_weights_biases(
        #     np.zeros([n_features, 1]), np.array([[4.0]])
        # )

        layers = [
            self.layer_dense,
            Activation_Sigmoid(),
        ]
        loss_fn = Loss_MeanSquaredError()

        super().__init__(layers, loss_fn)

    def train(self, x_data, y_data, n_baches=None, bach_size=8, epochs=128):
        losses = []

        if not n_baches:
            n_baches = x_data.shape[0] // bach_size

        for _ in range(n_baches):
            rand_ind = np.random.randint(len(x_data), size=bach_size)
            x_sample = x_data[rand_ind].reshape((-1, self.n_features))
            y_sample = y_data[rand_ind].reshape(-1, 1)
            loss = super().train(
                X=x_sample,
                y_true=y_sample,
                learning_rate=0.001,
                epochs=epochs,
                print_every=100,
            )
            losses.append(loss)

        return np.mean(losses)
