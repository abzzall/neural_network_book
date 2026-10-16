"""Check backpropagation against central differences (chapter 6)."""

import numpy as np


def max_relative_error(net, X, y, h=1e-6):
    """Largest relative difference between analytic and numerical gradients."""
    net.loss_value(X, y)
    net.backward()
    worst = 0.0
    for value, grad in net.params():
        analytic = grad.copy()
        for idx in np.ndindex(value.shape):
            old = value[idx]
            value[idx] = old + h
            up = net.loss_value(X, y)
            value[idx] = old - h
            down = net.loss_value(X, y)
            value[idx] = old
            numeric = (up - down) / (2 * h)
            denom = max(abs(numeric) + abs(analytic[idx]), 1e-12)
            worst = max(worst, abs(numeric - analytic[idx]) / denom)
    return worst
