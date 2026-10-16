"""Tests for the chapter 6 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_network_shape(load_script):
    m = load_script("mlp_digits")
    net = m.build()
    assert net.forward(np.zeros((3, 64))).shape == (3, 10)


def test_backpropagation_passes_the_gradient_check(load_script):
    m = load_script("mlp_digits")
    X_train, _, y_train, _ = m.load_split()
    assert m.gradient_check(m.build(), X_train, y_train) < 1e-5


def test_training_beats_softmax_regression(load_script):
    m = load_script("mlp_digits")
    X_train, X_test, y_train, y_test = m.load_split()
    net = m.train(m.build(), X_train, y_train)
    assert net.losses[-1] < 0.05 * net.losses[0]
    assert m.accuracy(net, X_test, y_test) > 0.9556      # chapter 5's softmax regression


def test_mistakes_are_consistent_with_accuracy(load_script):
    m = load_script("mlp_digits")
    X_train, X_test, y_train, y_test = m.load_split()
    net = m.train(m.build(), X_train, y_train)
    wrong = m.mistakes(net, X_test, y_test)
    assert len(wrong) == round((1 - m.accuracy(net, X_test, y_test)) * len(y_test))
    assert all(true != predicted for _, true, predicted in wrong)
