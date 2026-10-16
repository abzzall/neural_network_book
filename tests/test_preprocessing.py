"""nnbook.preprocessing gives the same numbers as scikit-learn."""

import numpy as np
from sklearn import impute, preprocessing as skp

from nnbook import preprocessing as pp

rng = np.random.default_rng(0)
X = rng.normal(10.0, 3.0, (50, 3))
X_new = rng.normal(10.0, 3.0, (7, 3))


def test_standard_scaler_matches_sklearn():
    ours = pp.StandardScaler().fit(X)
    theirs = skp.StandardScaler().fit(X)
    assert np.allclose(ours.transform(X_new), theirs.transform(X_new))
    assert np.allclose(ours.transform(X).mean(axis=0), 0.0)
    assert np.allclose(ours.transform(X).std(axis=0), 1.0)


def test_min_max_scaler_matches_sklearn():
    assert np.allclose(pp.MinMaxScaler().fit(X).transform(X_new),
                       skp.MinMaxScaler().fit(X).transform(X_new))


def test_median_imputer_matches_sklearn():
    Xm = X.copy()
    Xm[rng.random(Xm.shape) < 0.2] = np.nan
    ours = pp.MedianImputer().fit(Xm).transform(Xm)
    theirs = impute.SimpleImputer(strategy="median").fit_transform(Xm)
    assert np.allclose(ours, theirs)


def test_one_hot_matches_sklearn_and_ignores_unknown():
    train = np.array(["north", "south", "centre", "north"])
    new = np.array(["south", "east"])
    ours = pp.OneHotEncoder().fit(train).transform(new)
    theirs = skp.OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        train.reshape(-1, 1)).transform(new.reshape(-1, 1))
    assert np.array_equal(ours, theirs)
    assert ours[1].sum() == 0


def test_stratified_split_keeps_proportions():
    y = np.array([1] * 10 + [0] * 90)
    train, test = pp.train_test_split(100, test_size=0.2, seed=1, stratify=y)
    assert len(test) == 20 and len(set(train) & set(test)) == 0
    assert y[test].sum() == 2 and y[train].sum() == 8
