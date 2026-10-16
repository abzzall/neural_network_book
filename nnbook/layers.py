"""The dense (fully connected) layer (chapter 6).

W has shape (outputs, inputs), so a batch X of shape (n, inputs) gives
Z = X @ W.T + b, the batch formula of chapter 2.
"""

import numpy as np


class Dense:
    def __init__(self, n_in, n_out, rng):
        # Small random weights scaled by 1/sqrt(n_in); chapter 7 explains the choice.
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_out, n_in))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.params = ((self.W, self.dW), (self.b, self.db))

    def forward(self, X):
        self.X = X
        return X @ self.W.T + self.b

    def backward(self, grad):
        self.dW[...] = grad.T @ self.X       # one outer product per sample, summed
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W                  # gradient with respect to the input
