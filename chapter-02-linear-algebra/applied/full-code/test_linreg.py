"""Tests for the chapter 2 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_design_matrix_has_a_column_of_ones(load_script):
    m = load_script("linreg")
    X = m.design_matrix(np.array([1.0, 2.0, 3.0]))
    assert X.shape == (3, 2)
    assert np.array_equal(X[:, 0], np.ones(3))


def test_gradient_matches_central_differences(load_script):
    m = load_script("linreg")
    x, y = m.make_data()
    X, theta, h = m.design_matrix(x), np.array([0.5, -1.0]), 1e-6
    numeric = [(m.mse(X, y, theta + h * e) - m.mse(X, y, theta - h * e)) / (2 * h)
               for e in np.eye(2)]
    assert np.allclose(m.gradient(X, y, theta), numeric, rtol=1e-6)


def test_gradient_descent_reaches_the_normal_equation_solution(load_script):
    m = load_script("linreg")
    x, y = m.make_data()
    X = m.design_matrix(x)
    eta = 0.9 * m.learning_rate_bound(X)
    assert np.allclose(m.gradient_descent(X, y, eta, 2000), m.normal_equations(X, y), atol=1e-6)


def test_gradient_descent_diverges_above_the_bound(load_script):
    m = load_script("linreg")
    x, y = m.make_data()
    X = m.design_matrix(x)
    eta = 1.1 * m.learning_rate_bound(X)
    assert m.mse(X, y, m.gradient_descent(X, y, eta, 100)) > 1e6
