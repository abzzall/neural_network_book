"""Quantisation, FGSM, and the optional ONNX export."""

from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ONNX = all(importlib.util.find_spec(m) for m in ('onnx', 'onnxruntime', 'onnxscript'))


def run(script, cwd):
    return subprocess.run([sys.executable, str(HERE / script)], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=900).stdout.splitlines()


def test_quantise_worked_example(tmp_path):
    assert run('quantize_numpy.py', tmp_path) == [
        'scale 0.011765, zero point -43',
        'int8 codes: [-128, -43, -1, 127]',
        'dequantised: -1.0000 0.0000 0.4941 2.0000',
        'layer: 262144 bytes -> 65536 bytes, max error 0.00181 (half a step: 0.00181)']


def test_quantise_keeps_zero_exact(load_script):
    m = load_script('quantize_numpy')
    q, s, z = m.quantise(np.array([0.3, 1.7, 0.0, 0.9]))
    assert m.dequantise(q, s, z)[2] == 0.0                     # padding zeros stay zero


def test_fgsm_moves_by_epsilon_and_hurts_more_than_noise(load_script, tmp_path):
    m = load_script('fgsm_demo')
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 10))
    x, y = torch.full((4, 1, 8, 8), 0.5), torch.tensor([0, 1, 2, 3])
    adv = m.fgsm(model, x, y, 0.1)
    assert torch.allclose((adv - x).abs(), torch.full_like(x, 0.1))
    lines = run('fgsm_demo.py', tmp_path)
    attack = [float(l.split('accuracy ')[1].split()[0]) for l in lines]
    noise = [float(l.split(', ')[1].split()[0]) for l in lines]
    assert attack[0] == noise[0] > 0.97
    assert attack[2] < noise[2] - 0.3                       # epsilon 0.1


@pytest.mark.skipif(not ONNX, reason='optional: pip install -r requirements-deployment.txt')
def test_export_matches_pytorch(tmp_path):
    lines = [l for l in run('export_onnx.py', tmp_path) if l.startswith(('PyTorch', 'ONNX'))]
    accuracy = lines[0].split(': ')[1]
    assert lines[1].split('accuracy ')[1].startswith(accuracy)
    assert float(lines[1].split('difference| ')[1]) < 1e-4
    assert int(lines[2].split(': ')[1].split()[0]) < int(lines[1].split(': ')[1].split()[0])


def test_fgsm_picture(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'fgsm_picture.py'), str(tmp_path)],
                           cwd=tmp_path, capture_output=True, text=True, check=True,
                           timeout=600).stdout.splitlines()
    assert lines[0].startswith('test digit ') and ' after the attack' in lines[0]
    label, before, after = (int(w) for w in
                            [lines[0].split('label ')[1].split(',')[0],
                             lines[0].split('predicted ')[1].split()[0],
                             lines[0].split(', ')[2].split()[0]])
    assert label == before != after
    assert lines[1] == 'largest pixel change 0.10 of a range of 1'
    assert len(list(tmp_path.glob('*.dat'))) == 3
