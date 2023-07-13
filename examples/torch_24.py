import matplotlib.pyplot as plt
import numpy as np

from libs.idx import read_idx
from libs.mnist import get_mnist, plot_image


def run():
    imgs, labels, rows, cols = get_mnist(items_to_read=10)
    img = np.array(imgs[1]).reshape((rows, cols))
    plot_image(img)

    # examples per iteration
    total_data = len(imgs)
    batch = 100
    iterations = 3000

    epochs = 5
