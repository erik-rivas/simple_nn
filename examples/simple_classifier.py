import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from libs.helpers import generate_data
from models.simple_categorical_model import SimpleClassificationModel


def run():
    n_classes = 2
    n_features = 2
    n_samples = n_classes * 100
    X_train, X_test, y_train, y_test = generate_data(n_classes, n_features, n_samples)

    # One-hot encode the labels
    one_hot = OneHotEncoder(sparse_output=False)
    y_train = one_hot.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot.fit_transform(y_test.reshape(-1, 1))

    network = SimpleClassificationModel(
        n_features=n_features, n_hidden=n_classes, n_classes=n_classes
    )
    network.train(
        X_train,
        y_train,
        epochs=1,
        batch_size=5,
        learning_rate=0.01,
        verbose=1,
    )

    # network.evaluate()
    # network.predict()
    # network.plot_history()

    # # Plotting
    # plt.figure(figsize=(10, 7))

    # Plot training set
    # plt.subplot(1, 2, 2)
    # plt.scatter(
    #     X_train[:, 0],
    #     X_train[:, 1],
    #     c=y_train,
    #     edgecolors="k",
    #     cmap=plt.cm.Paired,
    # )
    # plt.title("Training set")
    # plt.show()

    # # # Plot test set
    # # plt.subplot(1, 2, 2)
    # # plt.scatter(X_test[:, 0], X_test[:, 1], c=np.argmax(network.forward(X_test), axis=1), edgecolors='k', cmap=plt.cm.Paired)
    # # plt.title('Test set predictions')

    # # plt.tight_layout()
    # plt.show()
