import matplotlib.pyplot as plt
import numpy as np

from libs.idx import read_idx
from models.mnist import MnistModel
from neural_network.estimators.logistic_regession_model import LogisticRegresionModel


def run():
    model = MnistModel()

    model.plot_dataset_sample(index=0)

    model.train(epochs=10, batch_size=128, learning_rate=0.01, verbose=1)
    model.evaluate()
    model.predict()
    model.plot_history()

    # model.plot_confusion_matrix()
    # model.plot_errors()
    # model.plot_weights()
    # model.plot_activations()
    # model.plot_gradients()
    # model.plot_weights_gradients()
    # model.plot_weights_gradients_3d()
    # model.plot_weights_gradients_3d_surface()
    # model.plot_weights_gradients_3d_contour()
