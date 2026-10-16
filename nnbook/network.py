"""A sequence of layers trained by gradient descent (chapter 6)."""

import numpy as np


class Network:
    def __init__(self, layers, loss):
        self.layers, self.loss = layers, loss

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self):
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        return [p for layer in self.layers for p in layer.params]

    def loss_value(self, X, y):
        return self.loss.forward(self.forward(X), y)

    def step(self, learning_rate):
        for value, grad in self.params():
            value -= learning_rate * grad

    def fit(self, X, y, learning_rate=0.1, epochs=100):
        """Full-batch gradient descent; chapter 7 adds mini-batches and better optimisers."""
        self.losses = []
        for _ in range(epochs):
            self.losses.append(self.loss_value(X, y))
            self.backward()
            self.step(learning_rate)
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
