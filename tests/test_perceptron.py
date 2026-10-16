import numpy as np

from nnbook import Perceptron


def test_learns_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, -1, -1, 1])
    model = Perceptron().fit(X, y)
    assert np.array_equal(model.predict(X), y)
    assert model.errors_per_epoch[-1] == 0


def test_cannot_learn_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, 1, 1, -1])
    model = Perceptron(max_epochs=50).fit(X, y)
    assert len(model.errors_per_epoch) == 50
    assert model.errors_per_epoch[-1] > 0


def test_separable_clouds():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.5, (30, 2)), rng.normal(3, 0.5, (30, 2))])
    y = np.array([-1] * 30 + [1] * 30)
    model = Perceptron().fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0
