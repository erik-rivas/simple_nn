import numpy as np
from matplotlib import pyplot as plt

from neural_network.estimators import LogisticRegresionModel


class TestLogisticRegressionModel:
    def test_simple_regression(self):
        x = np.linspace(0, 10, 101).reshape((-1, 1))
        y_true = x * 0.5 + 10 + np.random.uniform(-2, 2)
        print(y_true)

        nn = LogisticRegresionModel(n_features=2)

        # nn.train(X=x, y_true=y_true, learning_rate=0.01, epochs=1000, print_every=100)

        # x_pred = np.linspace(0, 10, 11).reshape((-1, 1))
        # y_pred = nn.forward(x_pred)

        # plt.plot(x, y_true, c="blue")
        # plt.plot(x_pred, y_pred, "o", c="red")

        # plt.show()
