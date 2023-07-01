# Let's first create the data
import numpy as np


def spiral_data(samples=100, classes=3):
    features = np.zeros((samples * classes, 2))
    labels = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )
        features[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        labels[ix] = class_number
    return features, labels
