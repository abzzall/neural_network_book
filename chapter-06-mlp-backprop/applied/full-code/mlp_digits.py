"""Chapter 6 applied exercise: a multilayer network from scratch on handwritten digits.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn. Same split as chapter 5.
"""

import numpy as np
from sklearn.datasets import load_digits

from nnbook import Dense, Network, ReLU, SoftmaxCrossEntropy, max_relative_error
from nnbook.preprocessing import StandardScaler, train_test_split


def load_split():
    X, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X[train]), scaler.transform(X[test]), y[train], y[test]


# Step 1: 64 inputs -> 32 ReLU units -> 10 scores.
def build(hidden=32, seed=0):
    rng = np.random.default_rng(seed)
    return Network([Dense(64, hidden, rng), ReLU(), Dense(hidden, 10, rng)],
                   SoftmaxCrossEntropy())


# Step 2: check backpropagation on a small batch before training.
def gradient_check(net, X, y):
    return max_relative_error(net, X[:5], y[:5])


# Step 3: full-batch gradient descent.
def train(net, X, y, learning_rate=0.5, epochs=300):
    return net.fit(X, y, learning_rate, epochs)


# Step 4: accuracy on a set of samples.
def accuracy(net, X, y):
    return float(np.mean(net.predict(X) == y))


# Step 5: the test samples the network gets wrong, as (index, true, predicted).
def mistakes(net, X, y):
    predicted = net.predict(X)
    return [(int(i), int(y[i]), int(predicted[i])) for i in np.flatnonzero(predicted != y)]
