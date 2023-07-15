import numpy as np

from neural_network.loss_functions import CategoricalCrossEntropy


class TestCategoricalCrossEntropy:
    def test_calculate(self):
        layer = CategoricalCrossEntropy()

        # test softmax output
        y_pred = np.array(
            [
                [0.1],
                [0.2],
                [0.7],
            ]
        )
        # test one-hot encoded labels
        y_true = np.array(
            [
                [0],
                [0],
                [1],
            ]
        )

        loss = layer.calculate(y_pred, y_true)

        assert np.isclose(loss, 0.3665, atol=CategoricalCrossEntropy.epsilon)

    def test_backward(self):
        layer = CategoricalCrossEntropy()

        # test softmax output
        y_pred = np.array(
            [
                [0],
                [0],
                [1],
            ]
        )
        # test one-hot encoded labels
        y_true = np.array(
            [
                [0],
                [0],
                [1],
            ]
        )

        dvalues = layer.backward(y_pred, y_true)

        # assert np.isclose(dvalues, 0, atol=CategoricalCrossEntropy.epsilon)
