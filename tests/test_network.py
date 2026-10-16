"""Backpropagation in nnbook agrees with central differences and learns XOR."""

import numpy as np
import pytest

from nnbook import (Dense, MeanSquaredError, Network, ReLU, Sigmoid, SoftmaxCrossEntropy,
                    Tanh, max_relative_error)


@pytest.mark.parametrize("activation", [ReLU, Tanh, Sigmoid])
def test_gradients_match_central_differences(activation):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    y = rng.integers(0, 3, 6)
    net = Network([Dense(4, 5, rng), activation(), Dense(5, 3, rng)], SoftmaxCrossEntropy())
    assert max_relative_error(net, X, y) < 1e-6


def test_mse_gradient_matches_central_differences():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(5, 2))
    net = Network([Dense(3, 4, rng), Tanh(), Dense(4, 2, rng)], MeanSquaredError())
    assert max_relative_error(net, X, Y) < 1e-6


def test_hidden_layer_learns_xor():
    rng = np.random.default_rng(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    net = Network([Dense(2, 4, rng), Tanh(), Dense(4, 2, rng)], SoftmaxCrossEntropy())
    net.fit(X, y, learning_rate=0.5, epochs=2000)
    assert np.array_equal(net.predict(X), y)
