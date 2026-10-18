"""Run every chapter 8 script as a reader would and check what it prints.

Numbers are compared with a relative tolerance: the unregularised high-degree
fits are so ill-conditioned that their last digits depend on the machine's
floating-point library. Well-conditioned values agree to 1e-4.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
NUMBER = r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?"


def run(script, *args):
    result = subprocess.run([sys.executable, script, *args], cwd=HERE,
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def same(lines, expected, rtol=1e-4, loose=()):
    assert len(lines) == len(expected)
    for i, (got, want) in enumerate(zip(lines, expected)):
        assert re.sub(NUMBER, "#", got) == re.sub(NUMBER, "#", want), got
        tol = 5e-2 if i in loose else rtol
        for a, b in zip(re.findall(NUMBER, got), re.findall(NUMBER, want)):
            assert abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12), (got, want)


def test_bias_variance(tmp_path):
    same(run("bias_variance.py", str(tmp_path)), [
        "degree  0: training MSE = 0.5500, validation MSE = 0.6187",
        "degree  1: training MSE = 0.2745, validation MSE = 0.2885",
        "degree  3: training MSE = 0.0186, validation MSE = 0.05794",
        "degree  5: training MSE = 0.0107, validation MSE = 0.04713",
        "degree  9: training MSE = 0.0088, validation MSE = 0.05982",
        "degree 15: training MSE = 0.0054, validation MSE = 295.6",
    ], loose=(5,))
    assert (tmp_path / "degree_errors.dat").exists()


def test_ridge_demo():
    same(run("ridge_demo.py"), [
        "lambda = 0       training MSE = 0.0072, validation MSE = 0.1575, largest |coefficient| = 1.37e+07",
        "lambda = 1e-06   training MSE = 0.0093, validation MSE = 0.08446, largest |coefficient| = 63",
        "lambda = 0.0001  training MSE = 0.0137, validation MSE = 0.04725, largest |coefficient| = 17.2",
        "lambda = 0.01    training MSE = 0.0739, validation MSE = 0.1046, largest |coefficient| = 5.72",
        "lambda = 1       training MSE = 0.2445, validation MSE = 0.2601, largest |coefficient| = 0.608",
    ], loose=(0,))


def test_cv_demo():
    same(run("cv_demo.py"), [
        "lambda = 1e-08   cross-validated MSE = 0.0714 (fold std 0.0440)",
        "lambda = 1e-06   cross-validated MSE = 0.0502 (fold std 0.0163)",
        "lambda = 0.0001  cross-validated MSE = 0.0464 (fold std 0.0184)",
        "lambda = 0.01    cross-validated MSE = 0.0844 (fold std 0.0515)",
        "lambda = 1       cross-validated MSE = 0.1650 (fold std 0.0849)",
    ], loose=(0,))


def test_metrics_demo():
    assert run("metrics_demo.py") == [
        "accuracy = 0.8500, precision = 0.8000, recall = 0.6667, F1 = 0.7273",
        "matches scikit-learn: True",
        "ROC-AUC = 0.9375 (scikit-learn 0.9375)",
    ]


def test_calibration_demo(tmp_path):
    same(run("calibration_demo.py", str(tmp_path)), [
        "logistic regression  accuracy = 0.7847, Brier = 0.1470, ECE = 0.0458",
        "network              accuracy = 0.8480, Brier = 0.1272, ECE = 0.1112",
        "network + Platt      accuracy = 0.8447, Brier = 0.1084, ECE = 0.0270",
    ], rtol=1e-3)
    for tag in ("raw", "platt"):
        rows = np.loadtxt(tmp_path / f"reliability_{tag}.dat", skiprows=1)
        assert np.isclose(rows[:, 0].sum(), 1.0)       # bin shares cover the test set


def test_compare_models():
    same(run("compare_models.py"), [
        "logistic 10-fold accuracy = 0.9672 +- 0.0117",
        "network  10-fold accuracy = 0.9738 +- 0.0090",
        "paired differences: mean = 0.0067, sd = 0.0123, network better in 7 folds, worse in 2",
        "paired t = 1.718, p = 0.120 (scipy: 0.120)",
        "test set: difference = 0.0044, 95% bootstrap interval [-0.0111, 0.0200]",
        "minimum useful gain = 0.0100, supported by interval: False",
        "digits only the network gets right: 7, only logistic: 5",
    ], rtol=1e-3)


def test_conformal_demo():
    same(run("conformal_demo.py"), [
        "test accuracy = 0.9644",
        "answer 100% most confident: error rate 0.0356",
        "answer 90% most confident: error rate 0.0099",
        "answer 80% most confident: error rate 0.0000",
        "alpha = 0.10: qhat = 0.1985, coverage = 0.9022, mean set size = 0.911, "
        "single-class sets = 0.911, empty sets = 0.089, lowest class coverage = 0.727",
        "  digits shifted one pixel: coverage = 0.2600",
        "alpha = 0.02: qhat = 0.7068, coverage = 0.9778, mean set size = 1.033, "
        "single-class sets = 0.967, empty sets = 0.000, lowest class coverage = 0.955",
        "  digits shifted one pixel: coverage = 0.3800",
    ], rtol=1e-3)


def test_conformal_uses_the_exact_rank_and_small_sample_boundary(load_script):
    m = load_script("conformal_demo")
    probabilities = np.column_stack((1 - np.arange(1, 448) / 448, np.arange(1, 448) / 448))
    labels = np.zeros(447, dtype=int)
    assert m.conformal_threshold(probabilities, labels, 0.1) == 404 / 448
    assert np.isinf(m.conformal_threshold(probabilities[:1], labels[:1], 0.01))
    assert m.prediction_sets(probabilities[:1], float('inf')).all()


def test_precision_recall_worked_example(load_script):
    m = load_script("pr_curve")
    from sklearn.metrics import average_precision_score
    labels = np.zeros(100, dtype=int)
    labels[[0, 2, 9, 19, 49]] = 1
    scores = np.linspace(1, 0, 100)
    precision, recall, ap = m.ranked_curve(labels, scores)
    assert np.allclose(precision[[0, 2, 9, 19, 49]], [1, 2/3, 3/10, 4/20, 5/50])
    assert np.allclose(recall[[0, 2, 9, 19, 49]], [.2, .4, .6, .8, 1])
    assert np.isclose(ap, average_precision_score(labels, scores))
    assert run("pr_curve.py")[0] == "prevalence 0.05, average precision 0.4533"
