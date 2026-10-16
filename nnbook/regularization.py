"""Dropout and batch normalisation layers (chapter 8).

Both behave differently in training and in evaluation; Network.fit switches
them with the `training` attribute.
"""

import numpy as np


class Dropout:
    """Inverted dropout: zero each unit with probability p, scale the rest by 1/(1-p)."""

    params = ()

    def __init__(self, p=0.5, seed=0):
        self.p, self.training, self.rng = p, True, np.random.default_rng(seed)

    def forward(self, A):
        if not self.training or self.p == 0.0:
            self.mask = None
            return A
        self.mask = (self.rng.random(A.shape) >= self.p) / (1.0 - self.p)
        return A * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


class BatchNorm:
    """Standardise each unit over the batch, then scale by gamma and shift by beta."""

    def __init__(self, n, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(n), np.zeros(n)
        self.dgamma, self.dbeta = np.zeros(n), np.zeros(n)
        self.params = ((self.gamma, self.dgamma), (self.beta, self.dbeta))
        self.running_mean, self.running_var = np.zeros(n), np.ones(n)
        self.momentum, self.eps, self.training = momentum, eps, True

    def forward(self, Z):
        if self.training:
            mean, var = Z.mean(axis=0), Z.var(axis=0)
            self.running_mean += self.momentum * (mean - self.running_mean)
            self.running_var += self.momentum * (var - self.running_var)
        else:
            mean, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.Z_hat = (Z - mean) * self.inv_std
        return self.gamma * self.Z_hat + self.beta

    def backward(self, grad):
        self.dgamma[...] = np.sum(grad * self.Z_hat, axis=0)
        self.dbeta[...] = np.sum(grad, axis=0)
        g = grad * self.gamma                       # gradient with respect to Z_hat
        n = len(g)
        return self.inv_std / n * (n * g - g.sum(axis=0) - self.Z_hat * np.sum(g * self.Z_hat, axis=0))
