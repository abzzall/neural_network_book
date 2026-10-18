"""Run the chapter's demonstrations and check the quantities they print."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def run(script, cwd):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=60)
    return result.stdout.splitlines()


def test_bptt_demo(tmp_path):
    assert run('bptt_demo.py', tmp_path) == [
        'hidden: 0.537050, -0.015736',
        'class 0 probabilities: 0.745376, 0.492132',
        'mean cross-entropy: 0.485701',
        'dWx: -0.339664, dWh: 0.264234, dbh: 0.398352',
        'input gradients: -0.046830, 0.246005',
        'initial-state gradient: -0.023415',
    ]


def test_gates_demo(tmp_path):
    assert run('gates_demo.py', tmp_path) == [
        'LSTM gate error below 1e-12: True', 'GRU gate error below 1e-12: True']


def test_gradient_paths(tmp_path, load_script):
    m = load_script('gradient_paths')
    for w in (0.5, 1., 1.5):
        assert m.derivative(w, 20) == pytest.approx(w ** 20)
    assert run('gradient_paths.py', tmp_path) == [
        'weight 0.5, 20-step state derivative: 9.53674e-07',
        'weight 1.0, 20-step state derivative: 1',
        'weight 1.5, 20-step state derivative: 3325.26',
        'gradient norm before clipping: 5.0',
        'gradient norm after clipping: 1.000000',
    ]
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3., 4.])
    before = torch.nn.utils.clip_grad_norm_([p], max_norm=1.)
    assert before == 5 and p.grad.norm() <= 1


def test_character_model(tmp_path):
    lines = run('char_rnn.py', tmp_path)
    initial, final = [float(line.split(': ')[1]) for line in lines[:2]]
    assert 0 < final < initial / 5
    assert lines[2].startswith('greedy sample: ')
    sample = lines[2].split(': ')[1]
    assert len(sample) == 48 and sample.startswith('r')
    assert set(sample) <= set('rain falls. sun shines. wind blows. ')


def test_forecast_picture(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'forecast_picture.py'), str(tmp_path)],
                           cwd=tmp_path, capture_output=True, text=True, check=True,
                           timeout=600).stdout.splitlines()
    assert lines[0].startswith('validation: seasonal MAE 1.7625, GRU MAE (seed 42) ')
    assert lines[1].startswith('test: seasonal MAE 1.6962, GRU MAE (seed 42) ')
    for line in lines:
        assert float(line.rsplit(' ', 1)[1]) < float(line.split('MAE ')[1].split(',')[0])
    assert (tmp_path / 'forecast.dat').is_file()
