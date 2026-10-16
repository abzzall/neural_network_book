"""Run examples as a reader would; GPU checks skip on CPU-only machines."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
SCRIPTS = ['device_demo', 'timing_demo', 'memory_transfer', 'amp_demo',
           'compile_demo', 'triton_add', 'profile_demo']


def run(name, cwd, cpu=False):
    env = os.environ.copy()
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return subprocess.run([sys.executable, str(HERE / f'{name}.py')],
                          cwd=cwd, env=env, capture_output=True, text=True,
                          check=True, timeout=180).stdout.splitlines()


@pytest.mark.parametrize('name', SCRIPTS)
def test_cpu_only_paths(name, tmp_path):
    lines = run(name, tmp_path, cpu=True)
    assert lines == (['device: cpu', 'matrix product: [[5.0, 14.0], [14.0, 50.0]]']
                     if name == 'device_demo' else ['SKIP: CUDA unavailable'])


def test_work_model(tmp_path, load_script):
    m = load_script('work_model')
    for n in (0, 1, 255, 256, 257, 1003):
        programs, masked = m.coverage(n, 256)
        assert programs * 256 - masked == n
        assert 0 <= masked < 256
    assert run('work_model', tmp_path) == [
        'programs: 4, masked offsets: 21',
        'vector-add FLOP per byte: 0.083333',
        'ideal matmul FLOP per byte: 85.333333']


@CUDA
@pytest.mark.parametrize('name', ['device_demo', 'memory_transfer', 'amp_demo',
                                 'compile_demo', 'triton_add'])
def test_gpu_scripts(name, tmp_path):
    lines = run(name, tmp_path)
    if name == 'device_demo':
        assert lines == ['device: cuda', 'matrix product: [[5.0, 14.0], [14.0, 50.0]]']
    elif name == 'memory_transfer':
        assert lines[0] == 'payload bytes: 4096'
        assert int(lines[1].split(': ')[1]) >= 4096
        assert lines[2:] == ['reserved covers allocated: True',
                             'live allocation released: True']
    elif name == 'amp_demo':
        assert lines == ['prediction dtype: torch.float16', 'loss dtype: torch.float32',
                         'parameters updated: True', 'finite unscaled gradients: True']
    elif name == 'compile_demo':
        assert lines == ['compiled values and gradients checked: 257']
    else:
        assert lines == ['Triton vector addition checked: 1003 elements']


@CUDA
def test_timing_and_profile(tmp_path):
    lines = run('timing_demo', tmp_path)
    assert all(float(line.split(': ')[1]) > 0 for line in lines[:2])
    assert lines[2] == 'CPU reference check: passed'
    lines = run('profile_demo', tmp_path)
    assert lines[-2:] == ['matrix multiplication recorded: True', 'finite output: True']


@CUDA
@pytest.mark.parametrize('size', [0, 1, 255, 256, 257, 1003, 65537])
def test_triton_boundaries(load_script, size):
    m = load_script('triton_add')
    generator = torch.Generator(device='cuda').manual_seed(17)
    x, y = (torch.randn(size, device='cuda', generator=generator) for _ in range(2))
    torch.testing.assert_close(m.add(x, y), x + y, rtol=0, atol=0)


@CUDA
def test_triton_rejects_unsupported_inputs(load_script):
    m = load_script('triton_add')
    x = torch.ones(8, device='cuda')
    for a, b in [(x[::2], x[::2]), (x, x[:4]), (x.double(), x.double()),
                 (x.reshape(2, 4), x.reshape(2, 4)), (x.cpu(), x.cpu())]:
        with pytest.raises(ValueError, match='contiguous CUDA float32 vectors'):
            m.add(a, b)
    with pytest.raises(ValueError, match='forward-only'):
        m.add(x.requires_grad_(), x)
