import numpy as np
from sklearn.datasets import make_blobs

from examples import tf_31, tf_32, tf_33, torch_24
from examples.simple import test_linnear_classifier, test_simple_regression, test_spiral
from models.mnist import MnistModel
from neural_network.estimators.logistic_regession_model import LogisticRegresionModel

if __name__ == "__main__":
    # test_simple_regression()
    # test_linnear_classifier()
    # test_spiral()

    # tf_31.run()
    # tf_32.run()
    # tf_33.run()
    # torch_24.run()

    model = MnistModel()
    model.train(epochs=10, batch_size=128, learning_rate=0.01, verbose=1)
    model.evaluate()
    model.predict()
    model.plot_history()
    model.plot_confusion_matrix()
    model.plot_errors()
    model.plot_weights()
    model.plot_activations()
    model.plot_gradients()
    model.plot_weights_gradients()
    model.plot_weights_gradients_3d()
    model.plot_weights_gradients_3d_surface()
    model.plot_weights_gradients_3d_contour()
