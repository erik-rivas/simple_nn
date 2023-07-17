import numpy as np
from colorama import Fore, Style
from colorama import init as colorama_init
from sklearn.datasets import make_blobs

from examples import mnist_test, simple_classifier, tf_31, tf_32, tf_33, torch_24
from examples.simple import test_linnear_classifier, test_simple_regression, test_spiral
from neural_network.estimators.logistic_regession_model import LogisticRegresionModel

if __name__ == "__main__":
    colorama_init()
    # test_simple_regression()
    # test_linnear_classifier()
    # test_spiral()

    # tf_31.run()
    # tf_32.run()
    # tf_33.run()
    # simple_classifier.run()
    mnist_test.run()

    print(f"\n\n{Fore.GREEN}This is {Fore.RED}done! 🏁🏁🏁{Style.RESET_ALL}!!!\n\n")
