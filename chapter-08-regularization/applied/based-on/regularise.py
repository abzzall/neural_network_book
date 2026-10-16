"""Chapter 8 applied exercise (starting point): overfit a network on 200 digits, then regularise it.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_digits

from nnbook import SGD, BatchNorm, Dense, Dropout, Network, ReLU, SoftmaxCrossEntropy
from nnbook.preprocessing import StandardScaler, train_test_split


def load_splits():
    """200 training images, the rest of the chapter 5 training rows for validation, and the test rows."""
    X, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
    small = np.random.default_rng(0).permutation(train)[:200]
    val = np.setdiff1d(train, small)
    scaler = StandardScaler().fit(X[small])
    return [(scaler.transform(X[i]), y[i]) for i in (small, val, test)]


# Step 1: a network with 256 hidden units, optionally with batch norm or dropout.
def build(dropout=0.0, batchnorm=False):
    raise NotImplementedError("Step 1: Dense(64, 256), optional BatchNorm(256), ReLU, optional Dropout, Dense(256, 10)")


# Step 2: five settings, each a network and the arguments for fit.
def settings(validation):
    raise NotImplementedError("Step 2: baseline, L2, dropout, early stopping, batch norm (see INSTRUCTIONS.md)")


# Step 3: train every setting for up to 200 epochs in batches of 32.
def train_all(train, val):
    raise NotImplementedError("Step 3: fit every setting for 200 epochs in batches of 32, seed 0")


def accuracy(net, data):
    X, y = data
    return float(np.mean(net.predict(X) == y))


# Step 4: choose on validation accuracy; only the chosen model is tested.
def choose(nets, val):
    raise NotImplementedError("Step 4: the name with the highest validation accuracy")
