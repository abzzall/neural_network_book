"""Activation layers (chapter 6). Each remembers its input in forward and
multiplies the incoming gradient by its own derivative in backward."""

import numpy as np

from nnbook.linear import sigmoid


class ReLU:
    params = ()

    def forward(self, Z):
        self.Z = Z
        return np.maximum(Z, 0.0)

    def backward(self, grad):
        return grad * (self.Z > 0)


class Tanh:
    params = ()

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, grad):
        return grad * (1.0 - self.A ** 2)


class Sigmoid:
    params = ()

    def forward(self, Z):
        self.A = sigmoid(Z)
        return self.A

    def backward(self, grad):
        return grad * self.A * (1.0 - self.A)
