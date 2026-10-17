"""Check group rates, gaps, the group threshold, and the measured audit."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import credit_data

DATA = credit_data.PATH.is_file()


def test_rates_and_gaps_by_hand(load_script):
    m = load_script('fairness_audit')
    approved = np.array([1, 1, 0, 1, 0, 0, 1, 1], bool)
    good = np.array([1, 0, 1, 1, 1, 0, 0, 1])
    young = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    r = m.rates(approved, good, young)
    # older: approve 3/4, TPR 2/3, FPR 1/1; young: approve 2/4, TPR 1/2, FPR 1/2
    assert r[0] == pytest.approx((0.75, 2 / 3, 1.0))
    assert r[1] == pytest.approx((0.5, 0.5, 0.5))
    assert m.gaps(r) == pytest.approx((-0.25, 0.5 - 2 / 3, 0.5))


def test_young_threshold_hits_the_target_rate(load_script):
    m = load_script('fairness_audit')
    p = np.linspace(0.01, 0.99, 200)
    good, young = np.ones(200, int), np.ones(200, int)
    t = m.young_threshold(p, good, young, 0.8)
    assert (p >= t).mean() == pytest.approx(0.8, abs=0.01)


@pytest.mark.skipif(not DATA, reason='run credit_data.py --download first')
def test_run_matches_reference(load_script):
    m = load_script('fairness_audit')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
