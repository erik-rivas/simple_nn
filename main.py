import numpy as np
from sklearn.datasets import make_blobs

from examples import simple_classifier, tf_31, tf_32, tf_33, torch_24
from examples.simple import test_linnear_classifier, test_simple_regression, test_spiral
from neural_network.estimators.logistic_regession_model import LogisticRegresionModel

if __name__ == "__main__":
    # test_simple_regression()
    # test_linnear_classifier()
    # test_spiral()

    # tf_31.run()
    # tf_32.run()
    # tf_33.run()
    simple_classifier.run()
