"""Chapter 8: an accurate but overconfident network, and Platt scaling."""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from nnbook.metrics import brier_score, expected_calibration_error, reliability

# 6000 samples, 5 % of labels flipped: no model can be certain.
X, y = make_classification(n_samples=6000, n_features=20, n_informative=6,
                           flip_y=0.05, class_sep=0.8, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=3000,
                                                    stratify=y, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, train_size=1500,
                                                stratify=y_rest, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train, X_cal, X_test = (scaler.transform(a) for a in (X_train, X_cal, X_test))

logistic = LogisticRegression().fit(X_train, y_train)
network = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=1000,
                        random_state=0).fit(X_train, y_train)   # no weight penalty


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# Platt scaling: a one-feature logistic regression on the network's logit,
# fitted on the calibration rows, never on the training or test rows.
platt = LogisticRegression().fit(logit(network.predict_proba(X_cal)[:, 1])[:, None], y_cal)
p_network = network.predict_proba(X_test)[:, 1]
probabilities = {"logistic regression": logistic.predict_proba(X_test)[:, 1],
                 "network": p_network,
                 "network + Platt": platt.predict_proba(logit(p_network)[:, None])[:, 1]}

if __name__ == "__main__":
    for name, p in probabilities.items():
        print(f"{name:20s} accuracy = {np.mean((p >= 0.5) == y_test):.4f}, "
              f"Brier = {brier_score(y_test, p):.4f}, "
              f"ECE = {expected_calibration_error(y_test, p):.4f}")
    if len(sys.argv) > 1:
        for name, tag in (("network", "raw"), ("network + Platt", "platt")):
            np.savetxt(Path(sys.argv[1]) / f"reliability_{tag}.dat",
                       reliability(y_test, probabilities[name]), fmt="%.6g",
                       header="share confidence frequency", comments="")
