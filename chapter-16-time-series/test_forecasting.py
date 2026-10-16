"""Unit checks of the baselines, metrics, autocorrelation and the TCN."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules


def test_acf_matches_definition(load_script):
    m = load_script('components')
    rng = np.random.default_rng(0)
    y = rng.normal(size=200).cumsum()
    r = m.acf(y, 5)
    d = y - y.mean()
    assert r[0] == pytest.approx(1)
    assert r[3] == pytest.approx(np.sum(d[3:] * d[:-3]) / np.sum(d * d))


def test_seasonal_naive_and_naive(load_script):
    m = load_script('baselines')
    y = np.arange(1, 15, dtype=float)                       # two weeks: 1..14
    assert m.seasonal_naive(y, 10).tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    assert m.naive(y, 3).tolist() == [14, 14, 14]
    assert m.moving_average(y, 2).tolist() == [11, 11]


def test_holt_winters_recovers_trend_plus_season(load_script):
    m = load_script('baselines')
    t = np.arange(140)
    season = np.array([3., -1, 0, 2, -2, 1, -3])
    y = 100 + 2 * t + season[t % 7]
    forecast, params = m.holt_winters(y, 14)
    expected = 100 + 2 * np.arange(140, 154) + season[np.arange(140, 154) % 7]
    # The simple start (level = mean of the first week) is centred mid-week, so a
    # small offset remains after 140 days; the shape is recovered exactly otherwise.
    np.testing.assert_allclose(forecast, expected, rtol=1e-4)
    np.testing.assert_allclose(np.diff(forecast), np.diff(expected), atol=0.01)


def test_vectorised_holt_winters_equals_separate_runs(load_script):
    m = load_script('baselines')
    y = np.random.default_rng(1).normal(50, 5, 60)
    a, b, g = np.array([0.2, 0.7]), np.array([0.1, 0.3]), np.array([0.5, 0.05])
    together = m.holt_winters_states(y, a, b, g)
    for i in range(2):
        alone = m.holt_winters_states(y, a[i], b[i], g[i])
        for x, z in zip(together, alone):
            np.testing.assert_allclose(x[i], z[0])


def test_metrics(load_script):
    m = load_script('metrics')
    y, f = np.array([10., 20.]), np.array([12., 15.])
    assert m.mae(y, f) == 3.5 and m.rmse(y, f) == pytest.approx(np.sqrt(14.5))
    assert m.mape(y, f) == pytest.approx(22.5)
    assert m.mase(y, f, np.array([1., 2, 3, 4]), m=2) == pytest.approx(3.5 / 2)
    assert m.pinball(y, f, 0.9) == pytest.approx((0.1 * 2 + 0.9 * 5) / 2)
    assert m.coverage(y, f - 3, f + 3) == 0.5


def test_tcn_is_causal_and_sees_its_receptive_field(load_script):
    m = load_script('tcn')
    assert m.receptive_field(3, [1, 2, 4, 8, 16]) == 63
    torch.manual_seed(0)
    model = m.TCN(horizon=2)
    first = torch.relu(model.first(torch.zeros(1, 1, 100)))
    x = torch.zeros(1, 1, 100, requires_grad=True)
    h = torch.relu(model.first(x))
    for block in model.blocks:
        h = h + torch.relu(block(h))
    h[0, :, 80].sum().backward()
    seen = torch.nonzero(x.grad[0, 0]).ravel()
    assert seen.max() == 80                                   # never a later input
    assert seen.min() >= 80 - 63 + 1                          # within the field
    assert first.shape == (1, 16, 100)


def test_holt_winters_one_step_matches_equations(load_script):
    m = load_script('baselines')
    y = np.array([10., 12, 14, 11, 13, 15, 18], float)       # m = 3: one update at t = 6
    y = y[:7]
    a, b, g = 0.5, 0.2, 0.3
    level0, trend0 = y[:3].mean(), (y[3:6].mean() - y[:3].mean()) / 3
    season = list(y[:3] - y[:3].mean())
    level, trend = level0, trend0
    for t in range(3, 7):
        s = season[t % 3]
        new_level = a * (y[t] - s) + (1 - a) * (level + trend)
        season[t % 3] = g * (y[t] - level - trend) + (1 - g) * s     # uses l_{t-1}, b_{t-1}
        trend = b * (new_level - level) + (1 - b) * trend
        level = new_level
    got_level, got_trend, got_season, _ = m.holt_winters_states(y, a, b, g, m=3)
    assert got_level[0] == pytest.approx(level) and got_trend[0] == pytest.approx(trend)
    np.testing.assert_allclose(got_season[0], season)
