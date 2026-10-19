"""nnbook.linear: stable activations, correct gradients, and agreement with scikit-learn."""

import numpy as np
import pytest
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


@pytest.mark.parametrize('model_class', [LogisticRegression, SoftmaxRegression])
def test_recorded_cross_entropy_does_not_cap_confident_errors(model_class):
    # The first update sends the correct class far below its competitor.
    # For a gap g, the true loss log(1 + exp(g)) is approximately g, not -log(eps).
    model = model_class(learning_rate=-1.0, epochs=2).fit(
        np.array([[100.0], [-100.0]]), np.array([1, 0]))
    gap = 5000.0 if model_class is LogisticRegression else 10000.0
    assert np.isfinite(model.losses).all()
    assert model.losses[1] == pytest.approx(gap)


@pytest.mark.parametrize('model_class', [LogisticRegression, SoftmaxRegression])
def test_recorded_loss_matches_the_update_gradient(model_class):
    X, y = np.array([[1.0, 2.0], [-2.0, 0.5], [0.5, -1.0]]), np.array([0, 1, 1])
    model = model_class(learning_rate=0.2, epochs=1).fit(X, y)
    # At zero weights p=1/2. Check the first update against an independent
    # central difference of the scalar cross-entropy objective.
    def objective(w):
        z = X @ w
        if model_class is LogisticRegression:
            return np.mean(np.logaddexp(0, z) - y * z)
        return np.mean(np.logaddexp(z[:, 0], z[:, 1]) - z[np.arange(len(y)), y])
    shape = model.w.shape if model_class is LogisticRegression else model.W.shape
    gradient = np.zeros(shape)
    for idx in np.ndindex(shape):
        delta = np.zeros(shape)
        delta[idx] = 1e-5
        gradient[idx] = (objective(delta) - objective(-delta)) / 2e-5
    weights = model.w if model_class is LogisticRegression else model.W
    np.testing.assert_allclose(weights, -0.2 * gradient, atol=1e-10)
    assert model.losses[0] == pytest.approx(objective(np.zeros(shape)))
