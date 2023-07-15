from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def calculate(self, y_pred, y_true):
        pass

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self, y_pred, y_true):
        pass
