"""Test threshold selection independently of optional pretrained weights."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_probe
from pretrained_probe import CACHE


def test_threshold_uses_calibration_labels():
    probabilities = np.array([0.1, 0.3, 0.4, 0.9])
    labels = np.array([0, 1, 1, 1])
    assert sentiment_probe.choose_threshold(probabilities, labels) == 0.2
    # A tie selects the candidate closest to the original 0.5 threshold.
    assert sentiment_probe.choose_threshold([0.01, 0.99], [0, 1]) == 0.5


def test_accuracy_and_errors():
    probabilities, labels = [0.1, 0.6, 0.8], [0, 0, 1]
    assert sentiment_probe.accuracy(probabilities, labels, 0.5) == pytest.approx(2 / 3)
    assert sentiment_probe.errors(['a', 'b', 'c'], probabilities, labels, 0.5) == ['b']


def test_original_dataset_is_disjoint():
    path = Path(__file__).resolve().parents[1] / 'reviews.json'
    data = json.loads(path.read_text())
    calibration, test = data['calibration'], data['test']
    shift = data['shift']
    assert len(calibration) == 8 and len(test) == 12 and len(shift) == 12
    texts = [{text for text, _ in part} for part in (calibration, test, shift)]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])
    assert {label for _, label in calibration + test + shift} == {0, 1}


@pytest.mark.skipif(not (CACHE / 'model.safetensors').is_file(), reason='download pinned model first')
def test_pretrained_experiment():
    results = sentiment_probe.run()
    assert results['threshold'] in [0.2, 0.35, 0.5, 0.65, 0.8]
    for name in ('calibration_accuracy', 'test_accuracy_before', 'test_accuracy_after'):
        assert 0 <= results[name] <= 1
    assert len(results['test_errors']) == round((1 - results['test_accuracy_after']) * 12)
    assert len(results['shift_errors']) == round((1 - results['shift_accuracy']) * 12)
    assert len(results['shift_error_scores']) == len(results['shift_errors'])
