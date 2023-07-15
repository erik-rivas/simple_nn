import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from models.simple_categorical_model import SimpleClassificationModel


def run():
    # Generate a binary classification dataset
    n_samples = 10
    n_features = 2
    n_classes = 3
    random_state = 101

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )

    # One-hot encode the labels
    one_hot = OneHotEncoder(sparse=False)
    y = one_hot.fit_transform(y.reshape(-1, 1))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Plot training set
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=np.argmax(y_train, axis=1),
        edgecolors="k",
        cmap=plt.cm.Paired,
    )
    plt.title("Training set")

    # network = SimpleClassificationModel(n_features=n_features, n_classes=n_classes)
    # network.train(X_train, y_train, epochs=1, batch_size=5, learning_rate=0.01)

    # network.evaluate()
    # network.predict()
    # network.plot_history()

    # # # Plotting
    # # plt.figure(figsize=(10, 7))

    # # # Plot test set
    # # plt.subplot(1, 2, 2)
    # # plt.scatter(X_test[:, 0], X_test[:, 1], c=np.argmax(network.forward(X_test), axis=1), edgecolors='k', cmap=plt.cm.Paired)
    # # plt.title('Test set predictions')

    # # plt.tight_layout()
    # plt.show()
