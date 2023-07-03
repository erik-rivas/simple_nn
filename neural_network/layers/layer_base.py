from abc import ABC, abstractmethod


class Layer:
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, gradients: np.ndarray):
        pass
