"""Print reference results or check the completed anomaly-detection results table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
scikit-learn 1.9.1, 17 October 2026). Counts must match exactly; rates may
differ slightly across platforms.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'test_items': 506,
    'test_anomalies': 183,
    'ae_recall': 0.306011,
    'ae_false_alarm': 0.0660475,
    'ae_precision': 0.716316,
    'ae_precision_at_1pct': 0.0437215,
    'pca_recall': 0.123862,
    'pca_false_alarm': 0.0515996,
    'pca_precision': 0.556588,
    'pca_precision_at_1pct': 0.0232948,
}
COUNTS = ('test_items', 'test_anomalies')


def matches(name, actual, expected):
    if name in COUNTS:
        return actual == expected
    return bool(np.isclose(actual, expected, rtol=0.1, atol=0.01))


def check(path):
    recorded = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 2:
            recorded[cells[0].strip('`')] = cells[1]
    ok = True
    for name, expected in REFERENCE.items():
        try:
            actual = float(recorded.get(name, 'nan'))
            good = np.isfinite(actual) and matches(name, actual, expected)
        except ValueError:
            good = False
        print(f"{'OK' if good else 'MISMATCH'} {name}: recorded {recorded.get(name)}")
        ok = ok and good
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    for name, value in REFERENCE.items():
        print(f'{name}: {value:.6g}')
