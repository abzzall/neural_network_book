"""INCOMPLETE exercise starter. Credit decisions under the documented costs of the German Credit data.

Approving a bad risk costs 5, rejecting a good one costs 1 (the data set's own
cost matrix). Policies are compared by the mean cost per applicant on test data,
averaged over five random splits.
"""

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'chapter-23-interpretability'))
from credit_data import load
from credit_model import model as logistic_model

COST_APPROVE_BAD, COST_REJECT_GOOD = 5.0, 1.0


def mean_cost(approve, good):
    """Step 1: average cost per applicant, eq. (25.1)."""
    raise NotImplementedError("Step 1: complete mean_cost")


def bayes_threshold():
    """Step 2: approve when P(good) exceeds this value, eq. (25.2)."""
    raise NotImplementedError("Step 2: complete bayes_threshold")


def boosting_model(columns):
    """Gradient boosting on the same one-hot encoded inputs as the logistic model."""
    logistic = logistic_model(columns)
    return make_pipeline(logistic[0], HistGradientBoostingClassifier(random_state=0))


def policies(X_train, y_train, X_test):
    """Step 3: approval decisions of four policies for the test applicants."""
    raise NotImplementedError("Step 3: complete policies")


def run(seeds=range(5)):
    X, good, _ = load()
    totals, approvals = {}, {}
    for seed in seeds:                                # Step 4: average over splits
        X_tr, X_te, y_tr, y_te = train_test_split(X, good, test_size=0.3, random_state=seed,
                                                  stratify=good)
        for name, approve in policies(X_tr, y_tr, X_te).items():
            totals.setdefault(name, []).append(mean_cost(approve, y_te))
            approvals.setdefault(name, []).append(approve.mean())
    out = {'bayes_threshold': bayes_threshold()}
    for name in totals:
        out[f'{name}_cost'] = float(np.mean(totals[name]))
        out[f'{name}_approved'] = float(np.mean(approvals[name]))
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
