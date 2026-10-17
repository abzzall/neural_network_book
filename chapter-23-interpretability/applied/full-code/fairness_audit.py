"""Audit a credit-scoring model for applicants aged 25 or less, then mitigate.

Approval means a predicted good credit risk. Three variants are compared: the
model with age, the model without age, and the model with a separate threshold
for young applicants chosen on validation data to equalise true-positive rates.
Every number is a mean over five random train/validation/test splits.
"""

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from credit_data import load
from credit_model import model


def rates(approved, good, young):
    """Step 1: approval rate, TPR and FPR for each group (0 = older, 1 = young)."""
    out = {}
    for g in (0, 1):
        m = young == g
        out[g] = (approved[m].mean(), approved[m & (good == 1)].mean(),
                  approved[m & (good == 0)].mean())
    return out


def gaps(r):
    """Step 2: young minus older, eq. (23.4): parity, opportunity, odds."""
    parity = r[1][0] - r[0][0]
    opportunity = r[1][1] - r[0][1]
    odds = max(abs(opportunity), abs(r[1][2] - r[0][2]))
    return parity, opportunity, odds


def young_threshold(p_val, good_val, young_val, target_tpr):
    """Step 3: the threshold that gives young good applicants the target TPR."""
    scores = np.sort(p_val[(young_val == 1) & (good_val == 1)])
    return float(np.quantile(scores, 1 - target_tpr))


def evaluate(seed, X, good, young):
    idx = np.arange(len(good))
    train, test = train_test_split(idx, test_size=0.3, random_state=seed, stratify=good)
    train, val = train_test_split(train, test_size=0.25, random_state=seed,
                                  stratify=good[train])
    variants = {}
    for name, keep in (('with_age', list(X.columns)),
                       ('without_age', [c for c in X.columns if c != 'age'])):
        fitted = model(keep).fit(X.iloc[train][keep], good[train])
        p_test = fitted.predict_proba(X.iloc[test][keep])[:, 1]
        variants[name] = p_test >= 0.5
        if name == 'with_age':                     # Step 4: a threshold for the young
            p_val = fitted.predict_proba(X.iloc[val][keep])[:, 1]
            older_good = (young[val] == 0) & (good[val] == 1)
            t = young_threshold(p_val, good[val], young[val], (p_val[older_good] >= 0.5).mean())
            variants['group_threshold'] = np.where(young[test] == 1, p_test >= t, p_test >= 0.5)
    out = {}
    for name, approved in variants.items():        # Step 5: measure on the test set
        r = rates(approved, good[test], young[test])
        out[name] = ((approved == good[test]).mean(), *r[0], *r[1], *gaps(r))
    return out


def run(seeds=range(5)):
    names = ('accuracy', 'older_approval', 'older_tpr', 'older_fpr', 'young_approval',
             'young_tpr', 'young_fpr', 'parity_gap', 'opportunity_gap', 'odds_gap')
    X, good, young = load()
    rows = [evaluate(s, X, good.to_numpy(), young.to_numpy()) for s in seeds]
    return {f'{variant}_{n}': float(np.mean([r[variant][k] for r in rows]))
            for variant in rows[0] for k, n in enumerate(names)}


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
