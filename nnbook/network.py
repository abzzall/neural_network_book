"""A sequence of layers trained by gradient descent (chapters 6 and 7)."""

import numpy as np

from nnbook.optimizers import SGD


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

    def fit(self, X, y, learning_rate=0.1, epochs=100, optimizer=None,
            batch_size=None, seed=0, schedule=None):
        """Train for a number of epochs.

        With batch_size=None every step uses the whole training set (chapter 6);
        otherwise each epoch visits the samples in a new random order, in
        mini-batches (chapter 7). losses[k] is the training loss at the start
        of epoch k. schedule(epoch) may set the learning rate of each epoch.
        """
        optimizer = optimizer or SGD(learning_rate)
        rng = np.random.default_rng(seed)
        self.losses = []
        for epoch in range(epochs):
            if schedule is not None:
                optimizer.lr = schedule(epoch)
            if batch_size is None:
                self.losses.append(self.loss_value(X, y))
                self.backward()
                optimizer.step(self.params())
                continue
            self.losses.append(self.loss_value(X, y))
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                self.loss_value(X[batch], y[batch])
                self.backward()
                optimizer.step(self.params())
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
