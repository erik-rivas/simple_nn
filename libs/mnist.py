from matplotlib import pyplot as plt

from libs.idx import read_idx


def get_mnist(items_to_read=10):
    imgs, shape = read_idx(
        path="data/mnist/train-images-idx3-ubyte", items_to_read=items_to_read
    )
    _, rows, cols = shape

    labels, shape = read_idx(
        "data/mnist/train-labels-idx1-ubyte", items_to_read=items_to_read
    )

    return imgs, labels, rows, cols


def plot_image(img):
    plt.imshow(img, cmap="gray")
    plt.show()
