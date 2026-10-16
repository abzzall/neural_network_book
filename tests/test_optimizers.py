"""nnbook.optimizers: each update matches its formula, and schedules behave."""

import math

import numpy as np

from nnbook.optimizers import SGD, AdaGrad, Adam, RMSProp, cosine, step_decay


def one_step(opt, value, grad):
    v, g = np.array([value]), np.array([grad])
    opt.step([(v, g)])
    return v[0]


def test_plain_sgd():
    assert math.isclose(one_step(SGD(lr=0.1), 1.0, 2.0), 0.8)


def test_momentum_accumulates_velocity():
    opt, v, g = SGD(lr=0.1, momentum=0.9), np.array([1.0]), np.array([1.0])
    opt.step([(v, g)])
    opt.step([(v, g)])
    assert math.isclose(v[0], 1.0 - 0.1 * 1.0 - 0.1 * 1.9)


def test_first_adam_step_has_size_lr():
    # After bias correction m_hat = g and v_hat = g^2, so the first step is lr * sign(g).
    assert math.isclose(one_step(Adam(lr=0.01), 1.0, 123.0), 0.99, rel_tol=1e-6)


def test_adamw_decays_weights_even_with_zero_gradient():
    assert math.isclose(one_step(Adam(lr=0.1, weight_decay=0.5), 2.0, 0.0), 2.0 - 0.1 * 0.5 * 2.0)


def test_adagrad_and_rmsprop_first_steps():
    assert math.isclose(one_step(AdaGrad(lr=0.1), 1.0, 4.0), 0.9, rel_tol=1e-6)
    assert math.isclose(one_step(RMSProp(lr=0.1, rho=0.9), 1.0, 4.0),
                        1.0 - 0.1 * 4.0 / math.sqrt(0.1 * 16.0), rel_tol=1e-6)


def test_schedules():
    assert step_decay(0.1, 25, drop=0.5, every=10) == 0.1 * 0.25
    assert cosine(1.0, 0, total=10, warmup=2) == 0.5
    assert cosine(1.0, 2, total=10, warmup=2) == 1.0
    assert abs(cosine(1.0, 10, total=10, warmup=2)) < 1e-12


def test_every_optimiser_minimises_a_quadratic():
    for opt in (SGD(0.1), SGD(0.05, momentum=0.9), AdaGrad(0.5), RMSProp(0.05), Adam(0.1)):
        x = np.array([3.0, -2.0])
        for _ in range(500):
            opt.step([(x, 2 * x)])
        assert np.linalg.norm(x) < 0.05, type(opt).__name__
