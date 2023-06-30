from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, dvalues):
        pass
