import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from libs.helpers import generate_data
from models.simple_categorical_model import SimpleClassificationModel


class SimpleClassifier:
    def __init__(self, n_classes, n_features, n_samples, random_state=75):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state

    def generate_data(self):
        X, y = generate_data(
            self.n_classes, self.n_features, self.n_samples, self.random_state
        )
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

    def generate_network(self, epochs=500, batch_size=8, learning_rate=0.01, verbose=1):
        # Create a model
        str_layers = (
            f"{self.n_features}::10_tanh,10::10_tanh,10::{self.n_classes}_softmax"
        )
        network = SimpleClassificationModel(str_layers)
        self.network = network

        network.train(
            self.X_train,
            self.y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # accuracy, precision, recall, f1_score = network.evaluate(self.X_test, self.y_test_onehot)
        self.y_pred = self.network.predict(self.X_test)

        return network

    def plot_results(self):
        # Plotting
        plt.figure(figsize=(10, 7))

        # Plot training set
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
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=plot_classes,
            edgecolors="k",
            cmap=plt.cm.Paired,
        )
        plt.title("Test set predictions")
        # plt.tight_layout()

        plt.subplot(2, 1, 2)
        self.network.plot_history(show=False)

        plt.show()

    def run(self, epochs=500, batch_size=8, learning_rate=0.01, verbose=1):
        self.generate_data()
        self.generate_network(
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, verbose=1
        )
        self.plot_results()


def run():
    simple_classifier = SimpleClassifier(
        n_classes=10, n_features=2, n_samples=1000, random_state=101
    )
    simple_classifier.run(
        epochs=128,
        batch_size=8,
        learning_rate=0.01,
        verbose=100,
    )
