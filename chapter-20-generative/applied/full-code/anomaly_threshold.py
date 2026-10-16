"""Anomaly detection by reconstruction error: digit 3 plays the role of a defect.

Models learn only from normal digits. The alarm threshold is the 95th percentile
of the error on held-out normal digits, so about 5% of normal items raise a
false alarm. Results are means over seeds 0, 1 and 2.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoencoder import pca_error, reconstruction_error, train

ANOMALY, QUANTILE = 3, 0.95


def split(seed):
    """Step 1: normal digits into train/validation/test; all anomalies go to test."""
    images, labels = load_digits(return_X_y=True)
    images = images / 16.0
    normal, odd = images[labels != ANOMALY], images[labels == ANOMALY]
    train_x, rest = train_test_split(normal, test_size=0.4, random_state=seed)
    val_x, test_normal = train_test_split(rest, test_size=0.5, random_state=seed)
    test_x = np.vstack([test_normal, odd])
    is_anomaly = np.r_[np.zeros(len(test_normal), bool), np.ones(len(odd), bool)]
    return train_x, val_x, test_x, is_anomaly


def threshold(val_scores, quantile=QUANTILE):
    """Step 2: the alarm level that normal validation items exceed 5% of the time."""
    return float(np.quantile(val_scores, quantile))


def rates(scores, level, is_anomaly):
    """Step 3: recall (anomalies caught), false-alarm rate, and precision."""
    alarm = scores > level
    recall = alarm[is_anomaly].mean()
    false_alarm = alarm[~is_anomaly].mean()
    precision = alarm[is_anomaly].sum() / max(alarm.sum(), 1)
    return float(recall), float(false_alarm), float(precision)


def precision_at(prevalence, recall, false_alarm):
    """Step 4: precision if anomalies made up `prevalence` of all items, eq. (20.11)."""
    caught = recall * prevalence
    return caught / (caught + false_alarm * (1 - prevalence))


def run():
    torch.set_num_threads(4)
    results = {'ae': [], 'pca': []}
    for seed in (0, 1, 2):
        train_x, val_x, test_x, is_anomaly = split(seed)
        ae = train(train_x, seed=seed)
        pca = PCA(n_components=8, random_state=0).fit(train_x)
        # Step 5: threshold on validation, then evaluate on the test set
        for name, score in (('ae', lambda x: reconstruction_error(ae, x)),
                            ('pca', lambda x: pca_error(pca, x))):
            recall, false_alarm, precision = rates(score(test_x), threshold(score(val_x)),
                                                   is_anomaly)
            results[name].append((recall, false_alarm, precision,
                                  precision_at(0.01, recall, false_alarm)))
    out = {'test_items': int(len(test_x)), 'test_anomalies': int(is_anomaly.sum())}
    for name, rows in results.items():
        r, f, p, p1 = np.mean(rows, axis=0)
        out.update({f'{name}_recall': r, f'{name}_false_alarm': f,
                    f'{name}_precision': p, f'{name}_precision_at_1pct': p1})
    return out


if __name__ == '__main__':
    print(json.dumps({k: float(v) for k, v in run().items()}, indent=1))
