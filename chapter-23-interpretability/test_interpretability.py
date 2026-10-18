"""Shapley values, permutation importance, and gradient attributions."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import credit_data

DATA = credit_data.PATH.is_file()


def run(script, cwd):
    return subprocess.run([sys.executable, str(HERE / script)], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=600).stdout.splitlines()


def test_shapley_worked_example(tmp_path):
    assert run('shapley_exact.py', tmp_path)[:2] == [
        'Shapley values: 2.0000 3.0000 3.0000',
        'sum 8.0000 = f(x) - f(baseline) = 8.0000']


def test_shapley_axioms(load_script):
    m = load_script('shapley_exact')
    rng = np.random.default_rng(0)
    w = rng.normal(size=4)
    x, b = rng.normal(size=4), rng.normal(size=4)
    phi = m.shapley(lambda z: w @ z, x, b)                     # linear: w_i (x_i - b_i)
    np.testing.assert_allclose(phi, w * (x - b))
    f = lambda z: np.tanh(z[0] * z[1]) + z[2] ** 2             # efficiency, dummy feature
    phi = m.shapley(lambda z: f(z), x, b)
    assert phi.sum() == pytest.approx(f(x) - f(b)) and phi[3] == pytest.approx(0)


def test_integrated_gradients_on_a_linear_model(load_script):
    m = load_script('integrated_gradients')
    model = torch.nn.Linear(3, 2)
    x = torch.tensor([1.0, -2.0, 0.5])
    ig = m.integrated_gradients(model, x, 1, steps=8)
    torch.testing.assert_close(ig, model.weight[1].detach() * x)   # exact for linear f
    assert torch.equal(m.saliency(model, x, 1), model.weight[1].detach().abs())


def test_integrated_gradients_script(tmp_path):
    lines = run('integrated_gradients.py', tmp_path)
    assert float(lines[0].split(': ')[1]) > 0.95
    assert float(lines[1].split('= ')[1]) < 0.1
    errors = [float(line.split('error ')[1]) for line in lines[2:6]]
    assert errors[-1] < errors[0] and errors[-1] < 0.01
    sal, ig = (float(v) for v in
               lines[6].replace(',', '').split(': ')[1].split()[1::3])
    assert abs(sal) < 0.3 and abs(ig) < 0.3                     # depends on the training


def test_missing_data_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        credit_data.load(tmp_path / 'absent')


@pytest.mark.skipif(not DATA, reason='run credit_data.py --download first')
def test_data_and_permutation_importance(tmp_path):
    assert run('credit_data.py', tmp_path) == [
        'applicants: 1000, features: 20, good credit: 700',
        'aged 25 or less: 190 applicants, share with good credit 0.579',
        'older than 25: 810 applicants, share with good credit 0.728']
    lines = run('permutation_importance.py', tmp_path)
    assert lines[0] == 'test AUC: 0.8263'
    assert lines[2].split()[0] == 'checking'
    assert float(lines[-1].split(': ')[1]) < 0.02


@pytest.mark.skipif(not DATA, reason='run credit_data.py --download first')
def test_proxy_check(tmp_path):
    lines = run('proxy_check.py', tmp_path)
    assert lines[0] == 'AUC for predicting the young group from the other 19 features: 0.804'
    assert [line.split(':')[0].strip() for line in lines[1:]] == [
        'housing', 'status_sex', 'dependants']


def test_attribution_maps(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'attribution_maps.py'), str(tmp_path)],
                           cwd=tmp_path, capture_output=True, text=True, check=True,
                           timeout=600).stdout.splitlines()
    assert [line.split(', positive')[0] for line in lines] == [
        'test image 0: label 2, predicted 2', 'test image 1: label 8, predicted 8',
        'test image 2: label 2, predicted 2']
    shares = [float(line.rsplit(' ', 1)[1]) for line in lines]
    assert shares == pytest.approx([0.770, 0.617, 0.818], abs=0.02)
    assert len(list(tmp_path.glob('*.dat'))) == 12
