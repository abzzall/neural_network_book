"""Check derivative agreement before training and the final fitting error."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_training


def test_cpu_fit():
    initial, final, grad = cuda_training.fit()
    assert 5 < initial < 20
    assert final < 1e-6
    assert grad.shape == (5,) and torch.isfinite(grad).all()
    # Independent analytic gradient at zero parameters.
    x, y = cuda_training.data()
    expected = torch.cat([(-2 * x.T @ y / len(x)).flatten(), -2 * y.mean().reshape(1)])
    torch.testing.assert_close(grad, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('amp,tolerance', [(False, 1e-5), (True, 0.01)])
def test_gpu_fit(amp, tolerance):
    _, _, reference = cuda_training.fit()
    _, final, actual = cuda_training.fit('cuda', amp)
    assert final < 1e-5
    assert (actual - reference).abs().max() < tolerance


def test_cpu_rejects_float16_exercise():
    with pytest.raises(ValueError, match='requires CUDA'):
        cuda_training.fit('cpu', amp=True)


def test_script(tmp_path):
    script = Path(__file__).with_name('cuda_training.py')
    output = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, check=True, timeout=60)
    values = json.loads(output.stdout)
    assert values['cpu_final_mse'] < 1e-6
    if torch.cuda.is_available():
        assert values['cuda_fp32_gradient_error'] < 1e-5
        assert values['cuda_amp_gradient_error'] < 0.01
