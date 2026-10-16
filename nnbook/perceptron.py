"""Rosenblatt perceptron (chapter 3)."""

import numpy as np


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    Labels are expected in {-1, +1}.
    """

    def __init__(self, learning_rate: float = 1.0, max_epochs: int = 100) -> None:
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.errors_per_epoch: list[int] = []

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []
        for _ in range(self.max_epochs):
            errors = 0
            for x_i, y_i in zip(X, y):
                if y_i * (x_i @ self.w + self.b) <= 0.0:  # misclassified
                    self.w += self.learning_rate * y_i * x_i
                    self.b += self.learning_rate * y_i
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:  # converged: all points classified correctly
                break
        return self
