"""Metrics against scikit-learn; dropout and batch norm against central differences."""

import numpy as np
from sklearn import metrics as skm

from nnbook import (BatchNorm, Dense, Dropout, Network, ReLU, SoftmaxCrossEntropy,
                    max_relative_error)
from nnbook.metrics import (brier_score, confusion_matrix, expected_calibration_error,
                            precision_recall_f1, r2, reliability, roc_auc)
from nnbook.model_selection import kfold

rng = np.random.default_rng(0)


def test_metrics_match_sklearn():
    y = rng.integers(0, 2, 200)
    scores = y + rng.normal(0, 0.8, 200)
    pred = (scores > 0.5).astype(int)
    assert np.array_equal(confusion_matrix(y, pred), skm.confusion_matrix(y, pred))
    p, r, f = precision_recall_f1(y, pred)
    assert np.isclose(p, skm.precision_score(y, pred))
    assert np.isclose(r, skm.recall_score(y, pred))
    assert np.isclose(f, skm.f1_score(y, pred))
    assert np.isclose(roc_auc(y, scores), skm.roc_auc_score(y, scores))
    t = rng.normal(size=50)
    t_hat = t + rng.normal(0, 0.3, 50)
    assert np.isclose(r2(t, t_hat), skm.r2_score(t, t_hat))


def test_kfold_covers_every_sample_once_and_stratifies():
    y = np.array([0] * 80 + [1] * 20)
    seen = []
    for train, val in kfold(100, k=5, seed=1, stratify=y):
        assert not set(train) & set(val) and len(train) + len(val) == 100
        assert y[val].sum() == 4
        seen.extend(val)
    assert sorted(seen) == list(range(100))


def test_batchnorm_and_dropout_gradients():
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
    net = Network([Dense(5, 6, rng), BatchNorm(6), ReLU(), Dense(6, 3, rng)], SoftmaxCrossEntropy())
    assert max_relative_error(net, X, y) < 1e-5            # batch norm in training mode
    drop = Dropout(0.5, seed=3)
    net = Network([Dense(5, 6, rng), ReLU(), drop, Dense(6, 3, rng)], SoftmaxCrossEntropy())
    drop.forward(np.ones((8, 6)))                          # draw one mask, then freeze it
    mask = drop.mask
    drop.forward = lambda A: A * mask
    assert max_relative_error(net, X, y) < 1e-6


def test_dropout_preserves_the_expected_activation():
    drop = Dropout(0.3, seed=0)
    A = np.ones((2000, 50))
    assert abs(drop.forward(A).mean() - 1.0) < 0.01
    drop.training = False
    assert np.array_equal(drop.forward(A), A)


def test_early_stopping_restores_the_best_parameters():
    X, y = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    Xv, yv = rng.normal(size=(30, 4)), rng.integers(0, 2, 30)   # labels unrelated: overfits fast
    net = Network([Dense(4, 32, rng), ReLU(), Dense(32, 2, rng)], SoftmaxCrossEntropy())
    net.fit(X, y, learning_rate=0.5, epochs=300, validation=(Xv, yv), patience=10)
    assert len(net.losses) < 300
    assert np.isclose(net.loss_value(Xv, yv), min(net.val_losses))


def test_calibration_metrics():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.05, 0.15, 0.95, 0.85, 0.55, 0.45, 0.65, 0.35])
    assert np.isclose(brier_score(y, p), skm.brier_score_loss(y, p))
    rows = reliability(y, p, bins=2)              # bins [0, 0.5) and [0.5, 1]
    assert np.allclose(rows, [[0.5, 0.25, 0.0], [0.5, 0.75, 1.0]])
    assert np.isclose(expected_calibration_error(y, p, bins=2), 0.25)
    perfect = np.repeat([0.2, 0.8], 10)
    outcome = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0])
    assert np.isclose(expected_calibration_error(outcome, perfect), 0.0)
