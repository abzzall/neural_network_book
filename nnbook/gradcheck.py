"""Check backpropagation against central differences (chapter 6)."""

import numpy as np


def max_relative_error(net, X, y, h=1e-5):
    """Largest relative difference between analytic and central-difference gradients.

    For float64, a step near the cube root of machine precision balances the
    second-order truncation error of the central difference against round-off.
    """
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
            # Near-zero gradients are checked in absolute terms: central-difference
            # noise is about machine epsilon over h (~1e-11 here), so an exactly-zero
            # analytic gradient can differ from the numerical one by that much, and
            # a relative error would report pure noise.
            diff = abs(numeric - analytic[idx])
            if diff > 1e-9:
                denom = max(abs(numeric) + abs(analytic[idx]), 1e-8)
                worst = max(worst, diff / denom)
    return worst
