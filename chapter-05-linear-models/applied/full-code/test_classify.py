"""Tests for the chapter 5 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_malignant_is_the_positive_class(load_script):
    m = load_script("classify")
    _, y, _ = m.load()
    assert y.sum() == 212 and len(y) == 569


def test_preparation_is_stratified_and_standardised(load_script):
    m = load_script("classify")
    X, y, _ = m.load()
    X_train, X_test, y_train, y_test = m.prepare(X, y)
    assert len(y_test) == 142 and y_test.sum() == 53
    assert np.allclose(X_train.mean(axis=0), 0.0) and np.allclose(X_train.std(axis=0), 1.0)


def test_training_reduces_the_loss_and_classifies_well(load_script):
    m = load_script("classify")
    X_train, X_test, y_train, y_test = m.prepare(*m.load()[:2])
    model = m.train(X_train, y_train)
    assert model.losses[-1] < 0.1 * model.losses[0]
    result = m.evaluate(model, X_test, y_test)
    assert result["accuracy"] > 0.95
    assert result["missed_malignant"] + result["false_alarms"] == round((1 - result["accuracy"]) * 142)


def test_largest_weights_are_sorted_by_size(load_script):
    m = load_script("classify")
    X, y, names = m.load()
    X_train, _, y_train, _ = m.prepare(X, y)
    weights = [abs(w) for _, w in m.largest_weights(m.train(X_train, y_train), names)]
    assert weights == sorted(weights, reverse=True) and len(weights) == 5
