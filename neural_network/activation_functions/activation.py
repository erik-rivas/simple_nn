from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass

    def update(self, learning_rate=None):
        pass
