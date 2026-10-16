"""Chapter 6 applied exercise (starting point): a multilayer network from scratch on handwritten digits.

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
    raise NotImplementedError("Step 1: Network([Dense(64, hidden), ReLU(), Dense(hidden, 10)], SoftmaxCrossEntropy())")


# Step 2: check backpropagation on a small batch before training.
def gradient_check(net, X, y):
    raise NotImplementedError("Step 2: max_relative_error on the first five training samples")


# Step 3: full-batch gradient descent.
def train(net, X, y, learning_rate=0.5, epochs=300):
    raise NotImplementedError("Step 3: net.fit with the given learning rate and epochs")


# Step 4: accuracy on a set of samples.
def accuracy(net, X, y):
    raise NotImplementedError("Step 4: the fraction of samples whose predicted class is correct")


# Step 5: the test samples the network gets wrong, as (index, true, predicted).
def mistakes(net, X, y):
    raise NotImplementedError("Step 5: (index, true, predicted) for every wrong prediction")
