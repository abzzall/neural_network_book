"""Print reference results or check the completed adaptation-comparison table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
transformers 5.20.1, 17 October 2026). Parameter counts must match exactly.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'encoder_parameters': 22713216,
    'full_trainable': 22713986,
    'lora_trainable': 74498,
    'probe_macro_f1': 0.908567,
    'full_macro_f1_mean': 0.964973,
    'lora_macro_f1_mean': 0.960363,
    'lora_macro_f1_min': 0.954313,
}
COUNTS = ('encoder_parameters', 'full_trainable', 'lora_trainable')


def matches(name, actual, expected):
    if name in COUNTS:
        return actual == expected
    return bool(abs(actual - expected) <= 0.015)


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
