"""Chapter 7 applied exercise (starting point): the same network trained with four optimisers.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn; the split of chapters 5 and 6.
"""

import numpy as np
from sklearn.datasets import load_digits

from nnbook import SGD, Adam, Dense, Network, ReLU, SoftmaxCrossEntropy
from nnbook.preprocessing import StandardScaler, train_test_split


def load_split():
    X, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X[train]), scaler.transform(X[test]), y[train], y[test]


def build(seed=0):
    rng = np.random.default_rng(seed)
    return Network([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)], SoftmaxCrossEntropy())


# Step 1: the four ways of training, as (optimiser, batch size).
def settings():
    raise NotImplementedError("Step 1: a dict name -> (optimiser, batch size) for the four settings")


# Step 2: train one network per setting for the same number of epochs.
def train_all(X, y, epochs=30):
    raise NotImplementedError("Step 2: build() and fit each setting for the same number of epochs")


# Step 3: the first epoch whose starting loss is below the target (None if never).
def epochs_to(losses, target=0.05):
    raise NotImplementedError("Step 3: the first epoch whose loss is below the target, or None")


# Step 4: test accuracy.
def accuracy(net, X, y):
    raise NotImplementedError("Step 4: the fraction of correct predictions")
