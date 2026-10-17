"""Monitor a deployed digit classifier for drift in its inputs.

Two quantities are monitored: each image's mean pixel intensity (an input
monitor) and the model's confidence, its largest class probability (an output
monitor). Reference: the training images. Five production scenarios are compared
with the reference by the population stability index (PSI) and the
Kolmogorov-Smirnov statistic.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from digit_model import load_images, shift_right, trained_model

PSI_ALARM, KS_C = 0.2, 1.628                 # c(alpha) for alpha = 0.01


def psi(reference, current, bins=10):
    """Step 1: sum over bins of (c - r) ln(c / r), bins from reference deciles, eq. (24.2)."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)    # avoid log(0)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_statistic(a, b):
    """Step 2: largest distance between the two empirical distribution functions, eq. (24.3)."""
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side='right') / len(a)
    fb = np.searchsorted(np.sort(b), grid, side='right') / len(b)
    return float(np.max(np.abs(fa - fb)))


def alarm(psi_value, ks_value, n_ref, n_cur):
    """Step 3: raise an alarm if PSI exceeds 0.2 or KS exceeds its 1% critical value."""
    critical = KS_C * np.sqrt((n_ref + n_cur) / (n_ref * n_cur))
    return bool(psi_value > PSI_ALARM or ks_value > critical)


def scenarios(X, y):
    swapped = y.clone()
    swapped[y == 1], swapped[y == 7] = 7, 1                  # the meaning of 1 and 7 changes
    low = y < 5
    return {'unchanged': (X, y), 'shifted one pixel': (shift_right(X), y),
            'fainter scans': (X * 0.6, y), 'only digits 0-4': (X[low], y[low]),
            'labels 1 and 7 swapped': (X, swapped)}


def run():
    torch.set_num_threads(1)
    model = trained_model()
    X_train, _, X_test, y_test = load_images()
    monitors = {'intensity': lambda X: X.mean(dim=(1, 2, 3)).numpy(),
                'confidence': lambda X: torch.softmax(model(X), 1).max(1).values.numpy()}
    with torch.inference_mode():
        reference = {m: f(X_train) for m, f in monitors.items()}
        out = {}
        for name, (X, y) in scenarios(X_test, y_test).items():   # Step 4: score each
            row = {'accuracy': (model(X).argmax(1) == y).float().mean().item()}
            for m, f in monitors.items():
                current = f(X)
                p, k = psi(reference[m], current), ks_statistic(reference[m], current)
                row[f'{m}_psi'], row[f'{m}_ks'] = p, k
                row[f'{m}_alarm'] = alarm(p, k, len(reference[m]), len(current))
            out[name] = row
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
