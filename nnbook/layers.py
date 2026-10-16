"""The dense (fully connected) layer (chapter 6).

W has shape (outputs, inputs), so a batch X of shape (n, inputs) gives
Z = X @ W.T + b, the batch formula of chapter 2.
"""

import numpy as np


class Dense:
    def __init__(self, n_in, n_out, rng, init="lecun"):
        # Random weights with a standard deviation chosen from the layer sizes
        # (chapter 7): "lecun" 1/sqrt(n_in), "glorot" sqrt(2/(n_in+n_out)),
        # "he" sqrt(2/n_in); a number is used as the standard deviation itself.
        std = {"lecun": 1.0 / np.sqrt(n_in), "glorot": np.sqrt(2.0 / (n_in + n_out)),
               "he": np.sqrt(2.0 / n_in)}.get(init, init)
        self.W = rng.normal(0.0, std, (n_out, n_in))
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
