"""Logistic and softmax regression from scratch (chapter 5).

Both are trained by full-batch gradient descent on the cross-entropy of
chapter 3. Their gradients have the same form as that of linear regression:
the features times the prediction errors, averaged over the samples.
"""

import numpy as np


def sigmoid(z):
    """1 / (1 + exp(-z)), computed without overflow for large |z|."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def softmax(Z):
    """Row-wise softmax; subtracting the row maximum avoids overflow."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


class LogisticRegression:
    """Binary classifier p(y=1|x) = sigmoid(w.x + b); labels in {0, 1}."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b, self.losses = np.zeros(d), 0.0, []
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            error = p - y                                # prediction errors
            self.w -= self.learning_rate * X.T @ error / n
            self.b -= self.learning_rate * error.mean()
            eps = 1e-12
            self.losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class SoftmaxRegression:
    """K-class classifier p(y=k|x) = softmax(x W + b)_k; labels in {0, ..., K-1}."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        K = int(y.max()) + 1
        Y = np.eye(K)[y]                                 # one-hot targets, shape (n, K)
        self.W, self.b, self.losses = np.zeros((d, K)), np.zeros(K), []
        for _ in range(self.epochs):
            P = softmax(X @ self.W + self.b)
            error = P - Y
            self.W -= self.learning_rate * X.T @ error / n
            self.b -= self.learning_rate * error.mean(axis=0)
            self.losses.append(-np.mean(np.log(P[np.arange(n), y] + 1e-12)))
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
