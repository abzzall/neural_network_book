"""Losses (chapter 6). forward returns the mean loss over the batch;
backward returns its gradient with respect to the network's output."""

import numpy as np

from nnbook.linear import softmax


class SoftmaxCrossEntropy:
    """Softmax of the scores followed by the cross-entropy; labels are integers."""

    def forward(self, Z, y):
        self.P, self.y = softmax(Z), y
        return -np.mean(np.log(self.P[np.arange(len(y)), y] + 1e-12))

    def backward(self):
        G = self.P.copy()
        G[np.arange(len(self.y)), self.y] -= 1.0      # P - Y
        return G / len(self.y)


class MeanSquaredError:
    def forward(self, Y_hat, Y):
        self.diff = Y_hat - Y
        return np.mean(np.sum(self.diff ** 2, axis=1))

    def backward(self):
        return 2.0 * self.diff / len(self.diff)
