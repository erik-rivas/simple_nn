import numpy as np


# generate Acuuracy class
class Accuracy:
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def __str__(self):
        return f"Accuracy: {self.accuracy}, Precision: {self.precision}, Recall: {self.recall}, F1 Score: {self.f1_score}"

    def calculate(self, y_pred, y_true):
        y_pred = np.round(y_pred)
        # calculate accuracy
        self.accuracy = np.mean(y_pred == y_true)

        # calculate precision
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        self.precision = true_positive / (true_positive + false_positive)

        # calculate recall
        fn = np.sum((y_pred == 0) & (y_true == 1))
        self.recall = true_positive / (true_positive + fn)

        # calculate f1_score
        self.f1_score = (
            2 * self.precision * self.recall / (self.precision + self.recall)
        )

        return self.accuracy, self.precision, self.recall, self.f1_score
