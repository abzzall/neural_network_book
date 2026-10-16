"""Check the split, threshold, rates, prevalence formula, and the measured results."""

import numpy as np
import pytest


def test_split_keeps_anomalies_out_of_training(load_script):
    m = load_script('anomaly_threshold')
    train_x, val_x, test_x, is_anomaly = m.split(0)
    assert len(train_x) + len(val_x) + (~is_anomaly).sum() == 1797 - 183
    assert is_anomaly.sum() == 183 and len(test_x) == len(is_anomaly)
    assert train_x.max() <= 1 and val_x.min() >= 0


def test_threshold_and_rates_by_hand(load_script):
    m = load_script('anomaly_threshold')
    assert m.threshold(np.arange(101.0)) == pytest.approx(95.0)
    scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 0.5])
    is_anomaly = np.array([False, False, False, True, True, True])
    recall, false_alarm, precision = m.rates(scores, 2.5, is_anomaly)
    assert (recall, false_alarm, precision) == pytest.approx((2 / 3, 1 / 3, 2 / 3))


def test_precision_at_prevalence(load_script):
    m = load_script('anomaly_threshold')
    # recall 0.5, false alarms 5%, 1% anomalies: 0.005 / (0.005 + 0.0495)
    assert m.precision_at(0.01, 0.5, 0.05) == pytest.approx(0.005 / 0.0545)
    assert m.precision_at(0.5, 1.0, 0.0) == 1.0


def test_run_matches_reference(load_script):
    m = load_script('anomaly_threshold')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
    assert results['ae_recall'] > results['pca_recall']
