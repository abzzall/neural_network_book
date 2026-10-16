"""Tests for the chapter 3 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_bernoulli_estimate_is_the_fraction_of_ones(load_script):
    m = load_script("estimate")
    assert m.bernoulli_mle(np.array([1.0, 0.0, 1.0, 1.0])) == 0.75


def test_bernoulli_gradient_descent_reaches_the_formula(load_script):
    m = load_script("estimate")
    clicks, _ = m.make_data()
    assert np.isclose(m.bernoulli_gd(clicks), m.bernoulli_mle(clicks), atol=1e-6)


def test_minimum_cross_entropy_is_the_entropy_of_the_data(load_script):
    m = load_script("estimate")
    clicks, _ = m.make_data()
    p = m.bernoulli_mle(clicks)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert np.isclose(m.bernoulli_nll(clicks, p), entropy)
    assert m.bernoulli_nll(clicks, p + 0.05) > m.bernoulli_nll(clicks, p)


def test_gaussian_estimates_divide_by_n(load_script):
    m = load_script("estimate")
    _, times = m.make_data()
    mu, var = m.gaussian_mle(times)
    assert np.isclose(mu, times.mean())
    assert np.isclose(var, times.var(ddof=0))


def test_gaussian_gradient_descent_reaches_the_formula(load_script):
    m = load_script("estimate")
    _, times = m.make_data()
    assert np.allclose(m.gaussian_gd(times), m.gaussian_mle(times), atol=1e-6)
    mu, var = m.gaussian_mle(times)
    assert m.gaussian_nll(times, mu, np.log(var)) < m.gaussian_nll(times, mu + 0.1, np.log(var))
