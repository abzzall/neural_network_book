"""Run every chapter 6 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script, *args):
    result = subprocess.run([sys.executable, str(HERE / script), *args],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_backprop_by_hand():
    assert run("backprop_by_hand.py") == [
        "z1 = [0.35 0.7 ]  a1 = [0.5866 0.6682]",
        "z2 = -0.2157, p = 0.4463, loss = 0.8068",
        "delta2 = -0.5537, dw2 = [-0.3248 -0.37  ] , db2 = -0.5537",
        "delta1 = [-0.1343  0.1841]",
        "dW1 = [[-0.1343, -0.0671], [0.1841, 0.0921]]",
    ]


def test_backprop_by_hand_matches_central_differences(load_script):
    m = load_script("backprop_by_hand")
    import numpy as np

    def loss(W1):
        a1 = m.sigmoid(W1 @ m.x + m.b1)
        p = m.sigmoid(m.w2 @ a1 + m.b2)
        return -np.log(p)

    h, numeric = 1e-6, np.zeros((2, 2))
    for idx in np.ndindex(2, 2):
        E = np.zeros((2, 2))
        E[idx] = h
        numeric[idx] = (loss(m.W1 + E) - loss(m.W1 - E)) / (2 * h)
    assert np.allclose(numeric, m.dW1, atol=1e-8)


def test_xor_mlp(tmp_path):
    assert run("xor_mlp.py", str(tmp_path)) == [
        "loss: first epoch 0.8017, last 0.0011",
        "predictions: [0 1 1 0]  targets: [0 1 1 0]",
        "hidden features: [[0.937, -0.918], [-0.949, -1.0], [1.0, 0.937], [0.946, -0.934]]",
    ]
    assert (tmp_path / "xor_hidden.dat").exists()


def test_approximation_demo():
    lines = run("approximation_demo.py")
    assert lines[:3] == [
        "hidden units =   1: mean squared error = 4.23e-01",
        "hidden units =   3: mean squared error = 3.76e-01",
        "hidden units =  10: mean squared error = 8.26e-04",
    ]
    for line in lines[3:]:                     # values below 1e-10 are rounding noise
        assert float(line.split("=")[-1]) < 1e-10


def test_output_heads():
    assert run("output_heads.py") == [
        "counts, identity + squared error:  18.6% of predictions negative, Poisson deviance 3.453",
        "counts, exp + Poisson likelihood:  0.0% of predictions negative, Poisson deviance 1.028",
        "  learned weights [ 0.529  0.998 -0.695], true [0.5, 1.0, -0.7]",
        "95% interval, one spread for all:  covers 100.0% where |x| < 1, 82.3% where |x| > 2",
        "95% interval, predicted spread:    covers 95.8% where |x| < 1, 96.2% where |x| > 2",
    ]
