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

    def set_training(self, flag):
        """Dropout and batch normalisation behave differently in training (chapter 8)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = flag

    def fit(self, X, y, learning_rate=0.1, epochs=100, optimizer=None,
            batch_size=None, seed=0, schedule=None, validation=None, patience=None):
        """Train for a number of epochs.

        With batch_size=None every step uses the whole training set (chapter 6);
        otherwise each epoch visits the samples in a new random order, in
        mini-batches (chapter 7). losses[k] is the training loss at the start
        of epoch k. schedule(epoch) may set the learning rate of each epoch.
        With validation=(X_val, y_val), val_losses are recorded too; with a
        patience, training stops when the validation loss has not improved for
        that many epochs and the best parameters are restored (chapter 8).
        """
        optimizer = optimizer or SGD(learning_rate)
        rng = np.random.default_rng(seed)
        self.losses, self.val_losses = [], []
        best, best_params, waited = np.inf, None, 0
        for epoch in range(epochs):
            if schedule is not None:
                optimizer.lr = schedule(epoch)
            self.set_training(False)
            self.losses.append(self.loss_value(X, y))
            if validation is not None:
                self.val_losses.append(self.loss_value(*validation))
                if patience is not None:
                    if self.val_losses[-1] < best:
                        best, waited = self.val_losses[-1], 0
                        best_params = [v.copy() for v, _ in self.params()]
                    else:
                        waited += 1
                        if waited >= patience:
                            break
            self.set_training(True)
            if batch_size is None:
                self.loss_value(X, y)
                self.backward()
                optimizer.step(self.params())
                continue
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                self.loss_value(X[batch], y[batch])
                self.backward()
                optimizer.step(self.params())
        if best_params is not None:
            for (value, _), saved in zip(self.params(), best_params):
                value[...] = saved
        self.set_training(False)
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
