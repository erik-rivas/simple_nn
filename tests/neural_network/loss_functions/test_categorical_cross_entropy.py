import numpy as np
import pytest

from neural_network.loss_functions import CategoricalCrossEntropy


class TestCategoricalCrossEntropy:
    @pytest.mark.parametrize(
        # fmt: off
        "y_pred, y_true, expected",
        [
            (
                np.array([[1, 0, 0],]),
                np.array([[1, 0, 0],]),
                0.0,
            ),
            (
                np.array([[0, 1, 0],]),
                np.array([[0, 1, 0],]),
                0.0,
            ),
            (
                np.array([[0, 0, 1],]),
                np.array([[0, 0, 1],]),
                0.0,
            ),
            (
                np.array([[0.1, 0.2, 0.7],]),
                np.array([[0.0, 0.0, 1.0],]),
                0.35667,
            ),
            (
                np.array([
                    np.array([[1, 0, 0],]),
                    np.array([[0, 1, 0],]),
                    np.array([[0, 0, 1],]),
                ]),
                np.array([
                    np.array([[1, 0, 0],]),
                    np.array([[0, 1, 0],]),
                    np.array([[0, 0, 1],]),
                ]),
                0.0,
            ),
            (
                np.array([
                    np.array([[1.0, 0.0, 0.0],]),
                    np.array([[0.0, 1.0, 0.0],]),
                    np.array([[0.0, 0.0, 1.0],]),
                    np.array([[0.1, 0.2, 0.7],]),
                ]),
                np.array([
                    np.array([[1, 0, 0],]),
                    np.array([[0, 1, 0],]),
                    np.array([[0, 0, 1],]),
                    np.array([[0, 0, 1],]),
                ]),
                0.089,
            ),
        ],
        # fmt: on
    )
    def test_calculate(self, y_pred, y_true, expected):
        layer = CategoricalCrossEntropy()

        loss = layer.calculate(y_pred, y_true)

        assert np.isclose(loss, expected, atol=0.001)

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
