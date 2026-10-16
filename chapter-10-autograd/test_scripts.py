"""Execute chapter examples as a reader would and verify their reported results."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def run(script, cwd):
    result = subprocess.run(
        [sys.executable, str(HERE / script)], cwd=cwd,
        capture_output=True, text=True, check=True, timeout=60,
    )
    return result.stdout.splitlines()


def test_value_demo(tmp_path):
    assert run('value_demo.py', tmp_path) == [
        'chapter 2: dL/dw = -4.0, dL/db = -4.0',
        'shared input: dz/dx = -0.007436 (PyTorch -0.007436), '
        'dz/dw1 = 0.004958 (PyTorch 0.004958)',
        'after two backward calls: dL/dw = -8.0',
    ]


def test_rules_demo(tmp_path):
    assert run('rules_demo.py', tmp_path) == [
        'leaf: True, result leaf: False',
        'hooked leaf gradient: [8.0, 12.0]',
        'retained gradient: [2.0, 2.0]',
        'accumulated gradient: [12.0, 18.0]',
        'no_grad result requires grad: False',
        'detached mutation: version mismatch detected',
    ]


def test_custom_function_script(tmp_path):
    lines = run('custom_function.py', tmp_path)
    assert lines[0] == 'gradcheck: True'
    assert lines[1].startswith('max gradient difference: ')
    assert float(lines[1].split(': ')[1]) < 1e-12


def test_custom_function_on_unseen_inputs(load_script):
    swish = load_script('custom_function').Swish.apply
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(3, 4, generator=generator, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(swish, (x,))
    torch.testing.assert_close(swish(x), x * torch.sigmoid(x))
    seed = torch.randn(3, 4, generator=generator, dtype=torch.double)
    actual, = torch.autograd.grad(swish(x), x, grad_outputs=seed)
    expected, = torch.autograd.grad(x * torch.sigmoid(x), x, grad_outputs=seed)
    torch.testing.assert_close(actual, expected)


def test_xor_training_script(tmp_path):
    lines = run('applied/full-code/autograd_xor.py', tmp_path)
    assert len(lines) == 4
    assert lines[0].startswith('initial MSE: ')
    assert lines[1].startswith('final MSE: ')
    assert lines[2].startswith('first gradient difference: ')
    assert lines[3].startswith('probabilities: ')
    initial, final, error = [float(line.split(': ')[1]) for line in lines[:3]]
    probabilities = np.array([float(v) for v in lines[3].split(': ')[1].split(', ')])
    targets = np.array([0., 1., 1., 0.])
    assert initial == pytest.approx(0.329236, abs=1e-6)
    assert final == pytest.approx(0.000420, abs=1e-6)
    assert 0 <= error < 1e-12
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    np.testing.assert_array_equal(probabilities >= 0.5, targets)
    assert np.mean((probabilities - targets) ** 2) == pytest.approx(final, abs=2e-6)
