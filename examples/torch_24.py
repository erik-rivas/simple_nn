import matplotlib.pyplot as plt
import numpy as np

from libs.idx import read_idx
from libs.mnist import get_mnist, plot_image
from neural_network.estimators.logistic_regession_model import LogisticRegresionModel


def run():
    imgs, labels, rows, cols = get_mnist(items_to_read=10)
    # img = np.array(imgs[1]).reshape((rows, cols))
    # plot_image(img)

    nn = LogisticRegresionModel(n_features=cols * rows)

    nn.train(
        x_data=imgs,
        y_data=labels,
        bach_size=8,
        learning_rate=0.01,
        epochs=1000,
    )

    # x_pred = np.linspace(0, 10, 11).reshape((-1, 1))
    # y_pred = nn.forward(x_pred)

    # plt.plot(x, y_true, c="blue")
    # plt.plot(x_pred, y_pred, "o", c="red")

    # plt.show()

    print("done!")
