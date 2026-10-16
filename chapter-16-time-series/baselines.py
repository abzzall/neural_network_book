"""Forecasting baselines: naive, seasonal naive, moving average, Holt-Winters."""

import numpy as np


def naive(y, h):
    return np.full(h, y[-1], dtype=float)


def seasonal_naive(y, h, m=7):
    """Repeat the last observed season, eq. (16.3)."""
    return np.array([y[len(y) - m + (i % m)] for i in range(h)], dtype=float)


def moving_average(y, h, window=7):
    return np.full(h, np.mean(y[-window:]))


def holt_winters_states(y, alpha, beta, gamma, m=7):
    """Additive Holt-Winters, eq. (16.4); parameters may be arrays (one run each)."""
    alpha, beta, gamma = (np.atleast_1d(np.asarray(p, float)) for p in (alpha, beta, gamma))
    level = np.full(alpha.shape, np.mean(y[:m]))
    trend = np.full(alpha.shape, (np.mean(y[m:2 * m]) - np.mean(y[:m])) / m)
    season = np.tile(np.asarray(y[:m], float) - np.mean(y[:m]), (len(alpha), 1))
    sse = np.zeros(alpha.shape)
    for t in range(m, len(y)):
        s = season[:, t % m]
        sse += (y[t] - (level + trend + s)) ** 2         # one-step-ahead error
        new_level = alpha * (y[t] - s) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season[:, t % m] = gamma * (y[t] - level - trend) + (1 - gamma) * s
        level = new_level
    return level, trend, season, sse


def holt_winters(y, h, m=7, grid=np.linspace(0.05, 0.95, 10)):
    """Fit the three smoothing parameters by grid search on one-step errors."""
    a, b, g = (p.ravel() for p in np.meshgrid(grid, grid, grid, indexing='ij'))
    *_, sse = holt_winters_states(y, a, b, g, m)
    best = np.argmin(sse)
    level, trend, season, _ = holt_winters_states(y, a[best], b[best], g[best], m)
    steps = np.arange(1, h + 1)
    phase = (len(y) + steps - 1) % m
    return level[0] + steps * trend[0] + season[0, phase], (a[best], b[best], g[best])
