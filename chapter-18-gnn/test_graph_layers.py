"""Numerical, symmetry and reader-invocation checks for graph layers."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def chapter_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))


def module(name):
    spec = importlib.util.spec_from_file_location('ch18_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_hand_calculation_and_permutation():
    m = module('gcn_numpy')
    adjacency, x, w = m.example()
    actual = m.normalise(adjacency) @ x @ w
    expected = [1 / 2 + 2 / np.sqrt(6),
                1 / np.sqrt(6) + 5 / 3,
                5 / 3 + 2 / np.sqrt(6), 3 / np.sqrt(6) + 1]
    np.testing.assert_allclose(actual[:, 0], expected)
    p = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(m.normalise(adjacency[p][:, p]) @ x[p] @ w, actual[p])
    np.testing.assert_allclose(m.normalise(np.zeros((2, 2))), np.eye(2))
    with pytest.raises(ValueError):
        m.normalise([[0, 1], [0, 0]])


def test_sparse_duplicates_and_gradient():
    m = module('gcn_torch')
    edges = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 2]])
    operator = m.sparse_operator(edges, 4, torch.float64)
    reference = module('gcn_numpy').normalise(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(operator.to_dense().numpy(), reference)
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: torch.sparse.mm(operator, z), (x,))


def test_attention_equivariance_mask_and_gradient():
    torch.manual_seed(42)
    layer = module('gat_torch').GraphAttention(2, 3).double()
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0],
                      [0, 1, 0, 0], [0, 0, 0, 0]])
    y, coefficients = layer(x, a)
    torch.testing.assert_close(coefficients.sum(1), torch.ones(4, dtype=x.dtype))
    assert coefficients[0, 2] == 0 and coefficients[3, 3] == 1
    p = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(layer(x[p], a[p][:, p])[0], y[p])
    assert torch.autograd.gradcheck(lambda z: layer(z, a)[0], (x,))


@pytest.mark.parametrize('name', ['gcn_numpy', 'gcn_torch', 'gat_torch',
                                 'graph_classification', 'link_prediction', 'smoothing_demo'])
def test_reader_invocation(name, tmp_path):
    result = subprocess.run([sys.executable, str(HERE / (name + '.py'))],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert not result.stderr
    expected = {'gcn_numpy': 'normalised adjacency:',
                'gcn_torch': 'dense and sparse agree: True',
                'gat_torch': 'non-neighbour weight: 0.0',
                'graph_classification': 'size-rule predictions: [0, 1]',
                'link_prediction': 'held-out edges absent from propagation: True',
                'smoothing_demo': 'depth 32: node variance'}
    assert expected[name] in result.stdout


def test_negative_sampling_excludes_all_observed_pairs():
    m = module('link_prediction')
    observed = [(1, 0), (2, 1)]
    negatives = m.negative_edges(4, observed, 4)
    assert len(set(negatives)) == 4
    assert all(u < v and (u, v) not in {(0, 1), (1, 2)} for u, v in negatives)
    with pytest.raises(ValueError):
        m.negative_edges(4, observed, 5)


def test_pooling_is_order_invariant_and_keeps_graphs_separate():
    m = module('graph_classification')
    x = torch.tensor([[1.], [3.], [20.]])
    assignment = torch.tensor([0, 0, 1])
    torch.testing.assert_close(m.pooled(x, assignment, 2), torch.tensor([[4.], [20.]]))
    p = torch.tensor([2, 0, 1])
    torch.testing.assert_close(m.pooled(x[p], assignment[p], 2, mean=True),
                               torch.tensor([[2.], [20.]]))


@pytest.mark.parametrize('relative', ['depth_experiment.py', 'applied/full-code/graph_exercise.py'])
def test_depth_experiment_as_reader(relative, tmp_path):
    result = subprocess.run([sys.executable, str(HERE / relative)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    runs = actual['runs'] if isinstance(actual, dict) else actual
    assert len(runs) == 15
    if isinstance(actual, dict):
        assert actual['selected_depth'] == 4
    reference = json.loads((HERE / 'applied/full-code/reference_results.json').read_text())
    for run, expected in zip(runs, reference):
        assert (run['seed'], run['depth']) == (expected['seed'], expected['depth'])
        assert abs(run['test_accuracy'] - expected['test_accuracy']) < .03


def test_depth_diagnosis(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'depth_diagnosis.py')], cwd=HERE,
                           capture_output=True, text=True, check=True,
                           timeout=600).stdout.splitlines()
    value = {line.split(' = ')[0]: float(line.split(' = ')[1]) for line in lines}
    assert value['depth 8: signal size at start'] < value['depth 4: signal size at start'] / 10
    assert value['depth 8: first-layer gradient norm'] < value['depth 4: first-layer gradient norm'] / 10
    assert abs(value['depth 8: training loss at end'] - 0.693) < 0.01    # never leaves log 2
    assert value['depth 4: training loss at end'] < 0.05
