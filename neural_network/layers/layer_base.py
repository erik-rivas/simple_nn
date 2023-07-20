from abc import ABC, abstractmethod

# from nptyping import NDArray
from numpy.typing import NDArray


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: NDArray):
        pass

    @abstractmethod
    def backward(self, gradients: NDArray):
        pass

    @abstractmethod
    def update(self, learning_rate: float):
        pass
