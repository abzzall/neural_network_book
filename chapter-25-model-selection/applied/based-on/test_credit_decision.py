"""Check the cost function, the Bayes threshold, and the measured comparison."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'chapter-23-interpretability'))
import credit_data


def test_cost_by_hand(load_script):
    m = load_script('credit_decision')
    approve = [1, 1, 0, 0, 1]
    good = [1, 0, 1, 0, 1]        # one bad approved (5), one good rejected (1)
    assert m.mean_cost(approve, good) == pytest.approx(6 / 5)


def test_bayes_threshold_minimises_expected_cost(load_script):
    m = load_script('credit_decision')
    t = m.bayes_threshold()
    assert t == pytest.approx(5 / 6)
    for p in (t - 0.01, t + 0.01):                 # expected cost of each action
        approve, reject = 5 * (1 - p), 1 * p
        assert (approve < reject) == (p > t)


@pytest.mark.skipif(not credit_data.PATH.is_file(), reason='download the credit data')
def test_run_matches_reference(load_script):
    m = load_script('credit_decision')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
