"""Run every chapter 3 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run(script):
    result = subprocess.run([sys.executable, str(HERE / script)],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_bayes_demo():
    assert run("bayes_demo.py") == [
        "P(positive) = 0.0585",
        "P(condition | positive) = 0.1538",
    ]


def test_stats_demo():
    assert run("stats_demo.py") == [
        "mean = [169.7   67.85]",
        "covariance = [[77.78, 66.21], [66.21, 93.6]]",
        "correlation = 0.7760",
        "matches NumPy: True",
        "correlation of x and x^2 on [-1, 1] = 0.0000",
    ]


def test_mle_demo():
    assert run("mle_demo.py") == [
        "heads = 7 of 10",
        "maximum-likelihood estimate k/n = 0.7000",
        "gradient descent on the cross-entropy: p = 0.7000",
        "cross-entropy at p = 0.5: 0.6931, at p = 0.7: 0.6109",
    ]


def test_entropy_demo():
    assert run("entropy_demo.py") == [
        "H(fair coin) = 1.0000 bits",
        "H(coin with p = 0.9) = 0.4690 bits",
        "H(p) = 1.1568, H(p, q) = 1.4219 bits",
        "KL(p || q) = 0.2651, KL(q || p) = 0.2771 bits",
    ]


def test_kl_is_non_negative_and_zero_only_for_equal_distributions(load_script):
    m = load_script("entropy_demo")
    rng = np.random.default_rng(0)
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
        assert m.kl(p, q) >= 0
    assert np.isclose(m.kl(m.p, m.p), 0.0)
