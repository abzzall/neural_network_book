"""Chapter 8 applied exercise: overfit a network on 200 digits, then regularise it.

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
    rng = np.random.default_rng(0)
    layers = [Dense(64, 256, rng)]
    if batchnorm:
        layers.append(BatchNorm(256))
    layers.append(ReLU())
    if dropout:
        layers.append(Dropout(dropout, seed=0))
    layers.append(Dense(256, 10, rng))
    return Network(layers, SoftmaxCrossEntropy())


# Step 2: five settings, each a network and the arguments for fit.
def settings(validation):
    sgd = lambda **kw: SGD(lr=0.05, momentum=0.9, **kw)  # noqa: E731
    return {
        "baseline": (build(), {"optimizer": sgd()}),
        "L2": (build(), {"optimizer": sgd(weight_decay=0.01)}),
        "dropout": (build(dropout=0.5), {"optimizer": sgd()}),
        "early stopping": (build(), {"optimizer": sgd(), "validation": validation, "patience": 10}),
        "batch norm": (build(batchnorm=True), {"optimizer": sgd()}),
    }


# Step 3: train every setting for up to 200 epochs in batches of 32.
def train_all(train, val):
    nets = {}
    for name, (net, kwargs) in settings(val).items():
        nets[name] = net.fit(*train, epochs=200, batch_size=32, seed=0, **kwargs)
    return nets


def accuracy(net, data):
    X, y = data
    return float(np.mean(net.predict(X) == y))


# Step 4: choose on validation accuracy; only the chosen model is tested.
def choose(nets, val):
    return max(nets, key=lambda name: accuracy(nets[name], val))
