"""nnbook.linear: stable activations, correct gradients, and agreement with scikit-learn."""

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogistic

from nnbook.linear import LogisticRegression, SoftmaxRegression, sigmoid, softmax

rng = np.random.default_rng(0)


def overlapping_classes(n=300, k=2, d=2):
    """Gaussian clouds that overlap, so the unregularised optimum is finite."""
    centres = rng.normal(0.0, 1.0, (k, d))
    y = rng.integers(0, k, n)
    return centres[y] + rng.normal(0.0, 1.0, (n, d)), y


def test_sigmoid_and_softmax_are_stable():
    assert np.allclose(sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0.0, 0.5, 1.0])
    P = softmax(np.array([[1000.0, 1000.0], [0.0, -1000.0]]))
    assert np.allclose(P, [[0.5, 0.5], [1.0, 0.0]])


def test_logistic_matches_sklearn_without_penalty():
    X, y = overlapping_classes()
    ours = LogisticRegression(learning_rate=0.5, epochs=5000).fit(X, y)
    theirs = SkLogistic(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(ours.w, theirs.coef_[0], atol=1e-4)
    assert np.isclose(ours.b, theirs.intercept_[0], atol=1e-4)


def test_softmax_matches_sklearn_probabilities():
    X, y = overlapping_classes(k=3)
    ours = SoftmaxRegression(learning_rate=0.5, epochs=8000).fit(X, y)
    theirs = SkLogistic(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)
    assert np.allclose(ours.predict_proba(X), theirs.predict_proba(X), atol=1e-4)


def test_logistic_loss_decreases():
    X, y = overlapping_classes()
    losses = LogisticRegression(epochs=200).fit(X, y).losses
    assert all(a >= b - 1e-12 for a, b in zip(losses, losses[1:]))
