"""Acceptance tests for the supplied capstone; none requires an MLP to win."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capstone_project as project
from capstone_acceptance import validate


@pytest.fixture(scope='module')
def completed():
    return project.run()


def test_partition_is_complete_and_independent_of_labels():
    splits = project.partition_ids(1200)
    assert [len(part) for part in splits] == [720, 240, 240]
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(1200))
    for a, b in zip(splits, project.partition_ids(1200)):
        np.testing.assert_array_equal(a, b)


def test_scaler_fits_only_training_rows():
    x, _ = project.synthetic_data()
    train, validation, test = project.partition_ids(len(x))
    scaler = project.fit_scaler(x, train)
    np.testing.assert_allclose(scaler.mean_, x[train].mean(axis=0))
    changed = x.copy()
    changed[np.concatenate([validation, test])] += 1000.
    again = project.fit_scaler(changed, train)
    np.testing.assert_array_equal(scaler.mean_, again.mean_)
    np.testing.assert_array_equal(scaler.scale_, again.scale_)


@pytest.mark.parametrize('actual,predicted', [([0, 0, 1, 1], [0, 1, 1, 1]),
    ([0, 1], [0, 0]), ([0, 0], [0, 0])])
def test_macro_f1(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    assert project.macro_f1(actual, predicted) == pytest.approx(
        f1_score(actual, predicted, labels=[0, 1], average='macro', zero_division=0))


def test_selection_prefers_declared_order_within_tolerance():
    values = {'linear': {'mean': .7}, 'quadratic': {'mean': .9},
              'mlp8': {'mean': .905}, 'mlp16': {'mean': .907}}
    assert project.choose_candidate(values) == 'quadratic'
    assert project.choose_candidate(values, tolerance=0.) == 'mlp16'


def test_error_rows_keep_stable_test_ids():
    assert project.error_rows([8, 3, 7], [0, 1, 0], [1, 1, 1]) == [
        {'row_id': 8, 'actual': 0, 'predicted': 1},
        {'row_id': 7, 'actual': 0, 'predicted': 1}]


def test_audit_rejects_bad_data():
    x, y = project.synthetic_data()
    splits = project.partition_ids(len(x))
    duplicate = x.copy()
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match='duplicate'):
        project.audit_data(duplicate, y, splits)
    missing = x.copy()
    missing[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        project.audit_data(missing, y, splits)
    with pytest.raises(ValueError, match='exactly once'):
        project.audit_data(x, y, (splits[0], splits[1], splits[0]))
    with pytest.raises(ValueError, match='binary'):
        project.audit_data(x, y + 2, splits)


def test_complete_evidence(completed):
    assert validate(completed)
    assert completed['audit']['rows'] == 1200
    assert completed['environment']['scikit_learn'] == '1.9.1'
    for section in ('validation', 'test'):
        assert all(0 <= item['mean'] <= 1 for item in completed[section].values())


def test_test_labels_cannot_change_selection(completed):
    x, y = project.synthetic_data()
    test = project.partition_ids(len(x))[2]
    changed = y.copy()
    changed[test] = 1 - changed[test]
    altered = project.run(x, changed)
    assert altered['chosen'] == completed['chosen']
    assert altered['validation'] == completed['validation']
    assert altered['scaler_mean'] == completed['scaler_mean']
    assert altered['data_sha256'] != completed['data_sha256']
    assert altered['test'] != completed['test']


@pytest.mark.parametrize('fault', ['overlap', 'chosen', 'mean', 'error_id'])
def test_acceptance_rejects_corrupt_evidence(completed, fault):
    bad = copy.deepcopy(completed)
    if fault == 'overlap':
        bad['split_ids']['test'][0] = bad['split_ids']['train'][0]
    elif fault == 'chosen':
        bad['chosen'] = 'linear' if bad['chosen'] != 'linear' else 'mlp16'
    elif fault == 'mean':
        bad['validation']['linear']['mean'] = float('nan')
    else:
        bad['representative_test_errors'] = [{'row_id': -1, 'actual': 0, 'predicted': 1}]
    with pytest.raises(ValueError, match='inconsistent'):
        validate(bad)


def test_project_script_from_any_directory(tmp_path):
    destination = tmp_path / 'results.json'
    script = Path(__file__).with_name('capstone_project.py')
    result = subprocess.run([sys.executable, str(script), '--output', str(destination)],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=90)
    assert json.loads(result.stdout) == json.loads(destination.read_text())
    assert validate(json.loads(destination.read_text()))
