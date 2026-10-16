"""Check the pinball loss, the inputs, the naive interval, and the measured comparison."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import bike_data

DATA = bike_data.PATH.is_file()


def test_pinball_loss_definition(load_script):
    m = load_script('quantile_bikes')
    y = torch.tensor([[10.0, 20.0]])
    q = torch.tensor([[[8.0, 10.0, 12.0], [25.0, 20.0, 15.0]]])
    # errors y - q: 2, 0, -2 and -5, 0, 5; levels 0.1, 0.5, 0.9
    expected = np.mean([0.1 * 2, 0, 0.1 * 2, 0.9 * 5, 0, 0.9 * 5])
    assert m.pinball_loss(y, q).item() == pytest.approx(expected)
    assert m.pinball_loss(y, torch.stack([y] * 3, dim=-1)).item() == 0


def test_design_adds_constant_and_calendar(load_script):
    m = load_script('quantile_bikes')
    X = m.design(np.ones((2, 56)), np.zeros((2, 14)))
    assert X.shape == (2, 71) and (X[:, 56] == 1).all()
    assert m.design(np.ones(56), np.zeros(14)).shape == (1, 71)


def test_naive_interval_on_a_repeating_week(load_script):
    m = load_script('quantile_bikes')
    week = np.array([1., 2, 3, 4, 5, 6, 7])
    q = m.naive_interval(np.tile(week, 60))
    assert q.shape == (14, 3)
    np.testing.assert_allclose(q, np.tile(week, 2)[:, None] * np.ones(3))


@pytest.mark.skipif(not DATA, reason='run bike_data.py --download first')
def test_run_matches_reference(load_script):
    m = load_script('quantile_bikes')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
    assert results['quantile_pinball'] < results['naive_pinball']
