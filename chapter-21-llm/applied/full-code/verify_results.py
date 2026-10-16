"""Print reference results or check the completed retrieval-augmented generation table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
transformers 5.20.1, 17 October 2026) with greedy decoding.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {'passages': 20, 'questions': 20, 'retrieval_hits': 20,
             'correct_without_retrieval': 1, 'correct_with_retrieval': 19}
TOLERANCE = {'passages': 0, 'questions': 0, 'retrieval_hits': 1,
             'correct_without_retrieval': 2, 'correct_with_retrieval': 2}


def matches(name, actual, expected):
    return bool(abs(actual - expected) <= TOLERANCE[name])


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
        print(f'{name}: {value}')
