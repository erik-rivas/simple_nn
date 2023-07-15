import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from libs.helpers import generate_data
from models.simple_categorical_model import SimpleClassificationModel


class SimpleClassifier:
    def __init__(self, n_classes, n_features, n_samples):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_samples = n_samples

    def generate_data(self):
        X, y = generate_data(self.n_classes, self.n_features, self.n_samples)
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        # One-hot encode the labels
        one_hot = OneHotEncoder(sparse_output=False)
        y_train_onehot = one_hot.fit_transform(y_train.reshape(-1, 1))
        y_test_onehot = one_hot.fit_transform(y_test.reshape(-1, 1))

        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        self.y_train_onehot = y_train_onehot
        self.y_test_onehot = y_test_onehot

    def generate_network(self):
        # Create a model
        network = SimpleClassificationModel(
            n_features=self.n_features,
            n_hidden=self.n_classes,
            n_classes=self.n_classes,
        )
        self.network = network

        network.train(
            self.X_train,
            self.y_train_onehot,
            epochs=5,
            batch_size=5,
            learning_rate=0.01,
            verbose=1,
        )

        network.evaluate(self.X_test, self.y_test_onehot)
        # network.predict()

        return network

    def plot_results(self):
        # Plotting
        plt.figure(figsize=(10, 7))

        # Plot training set
        plt.subplot(1, 2, 1)
        plt.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=self.y_train,
            edgecolors="k",
            cmap=plt.cm.Paired,
        )
        plt.title("Training set")

        y_pred = self.network.forward(self.X_test)
        # Plot test set
        plt.subplot(1, 2, 2)
        plt.scatter(
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=np.argmax(y_pred, axis=1),
            edgecolors="k",
            cmap=plt.cm.Paired,
        )
        plt.title("Test set predictions")

        plt.tight_layout()
        plt.show()
        self.network.plot_history()

    def run(self):
        self.generate_data()
        self.generate_network()
        self.plot_results()


def run():
    simple_classifier = SimpleClassifier(n_classes=3, n_features=2, n_samples=300)
    simple_classifier.run()
