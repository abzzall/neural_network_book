"""Run every chapter 12 script as a reader would and check what it prints."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent


def run(script):
    result = subprocess.run([sys.executable, script], cwd=HERE,
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_conv_numpy():
    assert run("conv_numpy.py") == [
        "output (3x3): [[3.0, 3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 0.0]]",
        "with padding 1: (5, 5)  with stride 2: (2, 2)",
        "matches torch.nn.functional.conv2d: True",
    ]


def test_kernel_gradient_matches_autograd(load_script):
    m = load_script("conv_numpy")
    rng = np.random.default_rng(0)
    image, kernel, upstream = rng.normal(size=(6, 6)), rng.normal(size=(3, 3)), rng.normal(size=(4, 4))
    K = torch.tensor(kernel, requires_grad=True)
    out = torch.nn.functional.conv2d(torch.tensor(image)[None, None], K[None, None])
    out.backward(torch.tensor(upstream)[None, None])
    assert np.allclose(m.kernel_gradient(image, upstream), K.grad.numpy())


def test_residual():
    assert run("residual.py") == [
        "input shape: (2, 8, 16, 16)  output shape: (2, 8, 16, 16)",
        "with F(x) = 0 the block passes relu(x) through: True",
    ]


def test_transfer_demo_shows_that_transfer_did_not_help_here():
    lines = run("transfer_demo.py")
    assert float(lines[0].split(":")[1]) > 0.95
    means = dict(re.findall(r"images\), (.+?)\s+(\d\.\d+) \+-", "\n".join(lines[1:])))
    means = {name: float(value) for name, value in means.items()}
    # Transfer did not help here: neither transferred variant beats training from
    # scratch by more than the spread between seeds (about 0.03).
    assert max(means["frozen features"], means["fine-tuned"]) <= means["from scratch"] + 0.03


def test_feature_maps(tmp_path):
    out = subprocess.run([sys.executable, "feature_maps.py", str(tmp_path)], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    # PyTorch training is not bit-identical across machines: the printed values are
    # those of the book's machine (0.9822; maps [15, 2, 9, 12]; 5 of 16 zero).
    assert abs(float(out[0].split("=")[1]) - 0.9822) < 0.01
    assert out[1] == "first test digit: label 1, predicted 1"
    assert out[2].startswith("feature maps with the largest total response: [")
    assert out[3].startswith("maps that are zero everywhere: ")
    assert len(list(tmp_path.glob("*.dat"))) == 9


sys.path.insert(0, str(HERE))
import kazakh_keywords  # noqa: E402


@pytest.mark.skipif(not (kazakh_keywords.FOLDER / 'data').is_dir(),
                    reason='run kazakh_keywords.py --download first')
def test_kazakh_keywords(tmp_path):
    out = subprocess.run([sys.executable, 'kazakh_keywords.py', '--figures', str(tmp_path)],
                         cwd=HERE, capture_output=True, text=True, check=True,
                         timeout=1800).stdout.splitlines()
    assert out[0] == ('2898 training and 725 test recordings of 35 commands, '
                      'spectrogram (40, 101)')
    assert out[1] == 'baseline on the time-averaged spectrum: accuracy 0.2441'
    # PyTorch training: the book's machine printed 0.9103, 0.6676, 0.3090 and 0.0648.
    accuracy = [float(line.rsplit(' ', 1)[1]) for line in out[2:6]]
    assert abs(accuracy[0] - 0.9103) < 0.03
    assert accuracy[0] > accuracy[1] > accuracy[2] > accuracy[3]
    assert out[6].startswith('most frequent confusion: ')
    assert len(list(tmp_path.glob('spectrogram_*.dat'))) == 2
