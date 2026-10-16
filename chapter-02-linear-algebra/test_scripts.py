"""Run every chapter 2 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run(script, *args):
    result = subprocess.run([sys.executable, str(HERE / script), *args],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_layer():
    assert run("layer.py") == [
        "z = [ 4.6 -0.2] shape (2,)",
        "Z = [[4.6, -0.2], [-2.9, 0.3]] shape (2, 2)",
    ]


def test_chain_rule():
    assert run("chain_rule.py") == [
        "forward:  u = 0.0, y_hat = 0.0, r = -2.0, L = 4.0",
        "backward: dL/dr = -4.0, dL/db = -4.0, dL/dw = -4.0",
    ]


def test_numerical_gradient():
    assert run("numerical_gradient.py") == [
        "numerical: [  1.  -12.5]",
        "analytic:  [  1.  -12.5]",
        "agree to 1e-6: True",
    ]


def test_numerical_gradient_on_another_function(load_script):
    m = load_script("numerical_gradient")
    x = np.array([0.3, -0.7, 1.1])
    estimate = m.numerical_gradient(lambda v: np.sin(v).sum(), x)
    assert np.allclose(estimate, np.cos(x), atol=1e-8)


def test_two_steps():
    assert run("two_steps.py") == [
        "step 0: theta = [0. 0.], loss = 15.0000",
        "step 1: gradient = [ -7.3333 -16.6667], theta = [0.7333 1.6667], loss = 0.2341",
        "step 2: gradient = [0.8    1.8222], theta = [0.6533 1.4844], loss = 0.0577",
        "normal equations: theta = [0.6667 1.5   ], loss = 0.0556",
        "learning-rate bound = 0.1803",
    ]


def test_gd_paths(tmp_path):
    assert run("gd_paths.py", str(tmp_path)) == [
        "learning-rate bound 2/lambda_max = 0.200",
        "eta = 0.05: f(start) = 51.75, f(after 20 steps) = 5.2047",
        "eta = 0.18: f(start) = 51.75, f(after 20 steps) = 0.0160",
        "eta = 0.205: f(start) = 51.75, f(after 20 steps) = 79.2041",
    ]
    for eta in (0.05, 0.18, 0.205):
        path = np.loadtxt(tmp_path / f"gd_eta_{eta}.dat", skiprows=1)
        assert path.shape == (21, 2)
        assert np.allclose(path[0], [-9.0, 1.5])
