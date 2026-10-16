"""Data preparation from scratch (chapter 4).

Each class learns its parameters with ``fit`` on the training data only and
applies them with ``transform``, the same interface as scikit-learn.
"""

import numpy as np


class StandardScaler:
    """Subtract each column's mean and divide by its standard deviation."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0      # a constant column stays constant
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


class MinMaxScaler:
    """Map each column linearly onto [0, 1] using the training minimum and maximum."""

    def fit(self, X):
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        self.span_ = np.where(span == 0, 1.0, span)
        return self

    def transform(self, X):
        return (X - self.min_) / self.span_


class MedianImputer:
    """Replace missing values (NaN) by the training median of their column."""

    def fit(self, X):
        self.median_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.median_, X)


class OneHotEncoder:
    """One column per category seen in training; unseen categories get all zeros."""

    def fit(self, values):
        self.categories_ = sorted(set(values))
        return self

    def transform(self, values):
        values = np.asarray(values)
        return np.column_stack([(values == c).astype(float) for c in self.categories_])


def train_test_split(n, test_size=0.2, seed=0, stratify=None):
    """Return shuffled index arrays (train, test); stratify keeps class proportions."""
    rng = np.random.default_rng(seed)
    if stratify is None:
        order = rng.permutation(n)
        n_test = round(n * test_size)
        return np.sort(order[n_test:]), np.sort(order[:n_test])
    train, test = [], []
    for label in np.unique(stratify):
        members = rng.permutation(np.flatnonzero(stratify == label))
        n_test = round(len(members) * test_size)
        test.extend(members[:n_test])
        train.extend(members[n_test:])
    return np.sort(train), np.sort(test)
