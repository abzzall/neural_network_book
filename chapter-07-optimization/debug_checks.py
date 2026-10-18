"""Chapter 7: three quick checks to run before a long training job."""

import warnings

import numpy as np
from sklearn.datasets import load_digits

from nnbook import SGD, Dense, Network, ReLU, SoftmaxCrossEntropy
from nnbook.preprocessing import StandardScaler, train_test_split

X, y = load_digits(return_X_y=True)                # pixels 0 to 16, ten classes
train, _ = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
X_raw, y_train = X[train], y[train]
X_std = StandardScaler().fit(X_raw).transform(X_raw)


def build(seed=0):
    rng = np.random.default_rng(seed)
    return Network([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)], SoftmaxCrossEntropy())


def final_loss(net, X, y, **fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # overflow is expected in check 3
        return net.fit(X, y, **fit).loss_value(X, y)


# Check 1: before training, the loss of K balanced classes should be near log K.
first_std = build().loss_value(X_std, y_train)
first_raw = build().loss_value(X_raw, y_train)

# Check 2: the network must be able to memorise a handful of samples.
tiny = final_loss(build(), X_std[:20], y_train[:20], optimizer=SGD(lr=0.5), epochs=300)

# Check 3: a learning rate far too large makes the loss rise instead of fall.
exploded = final_loss(build(), X_std, y_train, optimizer=SGD(lr=50.0), epochs=20)

if __name__ == "__main__":
    print(f"log 10 = {np.log(10):.4f}")
    print(f"initial loss, standardised inputs = {first_std:.4f}")
    print(f"initial loss, raw pixels          = {first_raw:.4f}")
    print(f"loss after memorising 20 samples  = {tiny:.6f}")
    print(f"loss with learning rate 50        = {exploded:.4f}")
