"""Check PSI and KS against definitions and SciPy, the alarm rule, and the scenarios."""

import numpy as np
import pytest
from scipy.stats import ks_2samp


def test_psi(load_script):
    m = load_script('drift_monitor')
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert m.psi(ref, ref) == pytest.approx(0.0)
    assert m.psi(ref, rng.normal(size=5000)) < 0.02
    assert m.psi(ref, rng.normal(1.0, 1.0, size=5000)) > 0.2


def test_ks_matches_scipy(load_script):
    m = load_script('drift_monitor')
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(0.3, 1.2, size=200)
    assert m.ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_alarm_rule(load_script):
    m = load_script('drift_monitor')
    assert m.alarm(0.25, 0.0, 1000, 1000) and not m.alarm(0.1, 0.05, 1000, 1000)
    assert m.alarm(0.0, 0.08, 1000, 1000)                     # critical value 0.0728


def test_run_matches_reference(load_script):
    m = load_script('drift_monitor')
    marker = load_script('verify_results')
    r = m.run()
    flat = {'unchanged_intensity_psi': r['unchanged']['intensity_psi'],
            'unchanged_confidence_alarm': r['unchanged']['confidence_alarm'],
            'shifted_accuracy': r['shifted one pixel']['accuracy'],
            'shifted_intensity_alarm': r['shifted one pixel']['intensity_alarm'],
            'shifted_confidence_alarm': r['shifted one pixel']['confidence_alarm'],
            'fainter_accuracy': r['fainter scans']['accuracy'],
            'fainter_intensity_psi': r['fainter scans']['intensity_psi'],
            'fainter_intensity_alarm': r['fainter scans']['intensity_alarm'],
            'swapped_accuracy': r['labels 1 and 7 swapped']['accuracy'],
            'swapped_confidence_alarm': r['labels 1 and 7 swapped']['confidence_alarm']}
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, float(flat[name]), expected), name
