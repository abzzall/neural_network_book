"""Chapter 8: abstaining on uncertain digits, and conformal prediction sets."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nnbook.preprocessing import train_test_split

X, y = load_digits(return_X_y=True)
rest, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)   # 450 test
order = np.random.default_rng(0).permutation(rest)
train, cal = order[:900], order[900:]                  # 900 to fit, 447 to calibrate
scaler = StandardScaler().fit(X[train])
model = LogisticRegression(max_iter=5000).fit(scaler.transform(X[train]), y[train])
p_cal = model.predict_proba(scaler.transform(X[cal]))
p_test = model.predict_proba(scaler.transform(X[test]))


def risk_at_coverage(p, labels, coverage):
    """Error rate on the most confident share `coverage` of the samples."""
    keep = np.argsort(-p.max(axis=1))[:int(round(coverage * len(labels)))]
    return float(np.mean(p[keep].argmax(axis=1) != labels[keep]))


def conformal_threshold(p, labels, alpha):
    """Split conformal: quantile of the scores 1 - p(true class) on calibration data."""
    scores = 1 - p[np.arange(len(labels)), labels]
    level = np.ceil((len(labels) + 1) * (1 - alpha)) / len(labels)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(p, qhat):
    return p >= 1 - qhat                               # boolean matrix: class k in the set


shifted = np.zeros_like(X[test]).reshape(-1, 8, 8)     # the test digits moved one pixel right
shifted[:, :, 1:] = X[test].reshape(-1, 8, 8)[:, :, :-1]
p_shift = model.predict_proba(scaler.transform(shifted.reshape(-1, 64)))

if __name__ == "__main__":
    print(f"test accuracy = {np.mean(p_test.argmax(1) == y[test]):.4f}")
    for coverage in (1.0, 0.9, 0.8):
        print(f"answer {coverage:.0%} most confident: error rate "
              f"{risk_at_coverage(p_test, y[test], coverage):.4f}")
    for alpha in (0.10, 0.02):
        qhat = conformal_threshold(p_cal, y[cal], alpha)
        sets = prediction_sets(p_test, qhat)
        covered = sets[np.arange(len(test)), y[test]]
        per_class = [covered[y[test] == k].mean() for k in range(10)]
        print(f"alpha = {alpha:.2f}: qhat = {qhat:.4f}, coverage = {covered.mean():.4f}, "
              f"mean set size = {sets.sum(1).mean():.3f}, "
              f"single-class sets = {np.mean(sets.sum(1) == 1):.3f}, "
              f"empty sets = {np.mean(sets.sum(1) == 0):.3f}, "
              f"lowest class coverage = {min(per_class):.3f}")
        moved = prediction_sets(p_shift, qhat)[np.arange(len(test)), y[test]]
        print(f"  digits shifted one pixel: coverage = {moved.mean():.4f}")
