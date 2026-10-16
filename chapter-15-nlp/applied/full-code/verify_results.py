"""Print reference results or check the completed spam-comparison results table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
transformers 5.20.1, scikit-learn 1.9.1, 16 October 2026). Counts must match
exactly; scores may differ slightly across platforms.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'labelled_messages': 200,
    'labelled_spam': 25,
    'test_messages': 1293,
    'tfidf_macro_f1': 0.888803,
    'frozen_macro_f1': 0.908567,
    'finetuned_macro_f1_mean': 0.964973,
    'finetuned_macro_f1_std': 0.00217104,
}
COUNTS = ('labelled_messages', 'labelled_spam', 'test_messages')


def matches(name, actual, expected):
    if name in COUNTS:
        return actual == expected
    if name.endswith('_std'):                    # spread of three seeds: loose check
        return abs(actual - expected) <= 0.01
    return bool(np.isclose(actual, expected, rtol=0, atol=0.01))


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
