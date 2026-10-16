"""Run every chapter 9 script as a reader would and check what it prints.

PyTorch does not promise bit-identical results across machines, so training
results are checked with a tolerance; exact arithmetic is checked exactly.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script, cwd=None):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_tensors_demo():
    assert run("tensors_demo.py") == [
        "shape: (2, 3)  dtype: torch.float32",
        "x + b broadcasts b over the rows: [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]",
        "x @ x.T = [[14.0, 32.0], [32.0, 77.0]]",
        "NumPy default dtype: float64 -> tensor dtype: torch.float64 first value: 99.0",
        "float32 epsilon: 1.1920928955078125e-07",
    ]


def test_autograd_reproduces_chapters_2_and_6():
    assert run("autograd_demo.py") == [
        "chapter 2: dL/dw = -4.0, dL/db = -4.0",
        "chapter 6: loss = 0.8068, dW1 = [[-0.1343, -0.0671], [0.1841, 0.0921]]",
    ]


def test_numpy_vs_torch():
    assert run("numpy_vs_torch.py") == [
        "loss: nnbook 1.0368419866, PyTorch 1.0368419866",
        "weight gradients agree to 1e-12: True",
    ]


def test_train_loop(tmp_path):
    lines = run("train_loop.py", cwd=tmp_path)
    accuracy = float(lines[0].split(":")[1])
    assert lines[0].startswith("test accuracy after 20 epochs:") and accuracy > 0.95
    assert lines[1] == "reloaded model gives identical outputs: True"
    assert (tmp_path / "outputs" / "digits_mlp.pt").exists()
