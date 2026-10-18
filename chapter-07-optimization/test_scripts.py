"""Run every chapter 7 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script, *args):
    result = subprocess.run([sys.executable, str(HERE / script), *args],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_optimizer_paths(tmp_path):
    assert run("optimizer_paths.py", str(tmp_path)) == [
        "gradient descent best lr = 0.07 f after 40 steps = 9.63e-02",
        "momentum         best lr = 0.1  f after 40 steps = 5.02e-03",
        "RMSProp          best lr = 0.3  f after 40 steps = 9.20e-04",
        "Adam             best lr = 1.0  f after 40 steps = 1.24e-01",
    ]
    assert len(list(tmp_path.glob("path_*.dat"))) == 4


def test_adam_by_hand():
    assert run("adam_by_hand.py") == [
        "step 1: g = 2.0000, m = 0.2000, v = 0.004000, m_hat = 2.0000, v_hat = 4.0000, t = 0.9000",
        "step 2: g = 1.8000, m = 0.3600, v = 0.007236, m_hat = 1.8947, v_hat = 3.6198, t = 0.8004",
        "step 3: g = 1.6008, m = 0.4841, v = 0.009791, m_hat = 1.7863, v_hat = 3.2671, t = 0.7016",
    ]


def test_adam_by_hand_matches_nnbook():
    import numpy as np
    from nnbook.optimizers import Adam
    t, opt = np.array([1.0]), Adam(lr=0.1)
    for _ in range(3):
        opt.step([(t, 2 * t)])
    assert np.isclose(t[0], 0.7016, atol=1e-4)


def test_init_demo(tmp_path):
    assert run("init_demo.py", str(tmp_path)) == [
        "tanh, std 0.01  std after layers 1, 5, 10: 0.1568, 0.0001, 0.0000; saturated at layer 10: 0%",
        "tanh, std 1     std after layers 1, 5, 10: 0.9748, 0.9740, 0.9740; saturated at layer 10: 86%",
        "tanh, Glorot    std after layers 1, 5, 10: 0.6283, 0.3210, 0.2230; saturated at layer 10: 0%",
        "ReLU, Glorot    std after layers 1, 5, 10: 0.5879, 0.1422, 0.0254",
        "ReLU, He        std after layers 1, 5, 10: 0.8315, 0.8044, 0.8118",
    ]


def test_debug_checks():
    lines = run("debug_checks.py")
    assert lines[:4] == [
        "log 10 = 2.3026",
        "initial loss, standardised inputs = 2.4224",
        "initial loss, raw pixels          = 5.3896",
        "loss after memorising 20 samples  = 0.000885",
    ]
    exploded = float(lines[4].split("=")[1])
    assert exploded > 5 * 2.3026                      # far above the starting loss
