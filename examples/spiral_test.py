import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from libs.helpers import spiral_data, train_test_split
from models.mnist import MnistModel
from models.simple_categorical_model import SimpleClassificationModel
from models.spiral_model import SimpleSpiralModel


class SpiralClassifier:
    def __init__(
        self, n_classes=2, n_features=2, n_hidden=64, n_samples=100, random_state=101
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.random_state = random_state

    def generate_data(self, n_samples):
        X, y = spiral_data(n_samples, self.n_classes)
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        # One-hot encode the labels
        one_hot = OneHotEncoder(sparse_output=False)
        y_train_onehot = one_hot.fit_transform(y_train.reshape(-1, 1))
        y_test_onehot = one_hot.fit_transform(y_test.reshape(-1, 1))

        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_onehot = y_train_onehot
        self.y_test_onehot = y_test_onehot

    def generate_network(
        self, epochs, batch_size, iter_per_batch, learning_rate, verbose
    ):
        # Create a model
        network = SimpleSpiralModel(n_classes=self.n_classes)
        self.network = network

        network.train(
            self.X_train,
            self.y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            iter_per_batch=iter_per_batch,
            learning_rate=learning_rate,
            verbose=verbose,
            live_plot=True,
        )

        accuracy, precision, recall, f1_score = network.evaluate(
            self.X_test, self.y_test_onehot
        )
        print(
            f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}"
        )
        self.y_pred = self.network.predict(self.X_train)

        return network

    def plot_results(self):
        # Plotting
        plt.figure(figsize=(10, 7))

        # # Plot training set
        plt.subplot(2, 2, 1)
        plt.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=self.y_train,
            edgecolors="k",
            cmap=plt.cm.Paired,
        )
        plt.title("Training set")

        if self.y_pred.shape[1] == 1:
            plot_classes = np.round(self.y_pred).astype(int)
        else:
            plot_classes = np.argmax(self.y_pred, axis=1)

        # Plot test set
        plt.subplot(2, 2, 2)
        plt.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=plot_classes,
            edgecolors="k",
            cmap=plt.cm.Paired,
        )
        plt.title("Test set predictions")
        # plt.tight_layout()

        plt.subplot(3, 1, 3)
        self.network.plot_history(show=False)

        plt.show()

    def run(
        self, n_samples, epochs, batch_size, iter_per_batch, learning_rate, verbose
    ):
        self.generate_data(n_samples)
        self.generate_network(
            epochs=epochs,
            batch_size=batch_size,
            iter_per_batch=iter_per_batch,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        self.plot_results()


def run():
    spiral_classifier = SpiralClassifier(
        n_classes=3, n_features=2, n_hidden=64, n_samples=1000, random_state=101
    )
    spiral_classifier.run(
        n_samples=1000,
        epochs=10000,
        batch_size=10,
        iter_per_batch=1,
        learning_rate=0.1,
        verbose=1000,
    )
