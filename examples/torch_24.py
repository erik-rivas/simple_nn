import matplotlib.pyplot as plt
import numpy as np

from libs.idx import read_idx


def run():
    imgs, shape = read_idx(path="data/mnist/train-images-idx3-ubyte", items_to_read=10)
    img = imgs[0]
    _, rows, cols = shape

    img = np.array(imgs[0]).reshape((rows, cols))
    plt.imshow(img, cmap="gray")
    plt.show()

    labels, shape = read_idx("data/mnist/train-labels-idx1-ubyte", items_to_read=10)
