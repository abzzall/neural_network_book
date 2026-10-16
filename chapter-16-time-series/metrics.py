"""Point and probabilistic forecast errors."""

import numpy as np


def mae(y, f):
    return np.mean(np.abs(np.asarray(y) - f))


def rmse(y, f):
    return np.sqrt(np.mean((np.asarray(y) - f) ** 2))


def mape(y, f):
    y = np.asarray(y, float)
    return 100 * np.mean(np.abs((y - f) / y))


def mase(y, f, history, m=7):
    """MAE scaled by the in-sample seasonal-naive MAE, eq. (16.5)."""
    history = np.asarray(history, float)
    scale = np.mean(np.abs(history[m:] - history[:-m]))
    return mae(y, f) / scale


def pinball(y, q, tau):
    """Quantile (pinball) loss for quantile level tau, eq. (16.7)."""
    diff = np.asarray(y) - q
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))


def coverage(y, lower, upper):
    y = np.asarray(y)
    return np.mean((y >= lower) & (y <= upper))
