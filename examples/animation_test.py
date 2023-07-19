import matplotlib.pyplot as plt
import numpy as np


def run():
    x = 0
    for i in range(100):
        x = x + 0.04
        y = np.sin(x)
        plt.scatter(x, y)
        plt.title("Real Time plot")
        plt.xlabel("x")
        plt.ylabel("sinx")
        plt.pause(0.05)

    # plt.show()
