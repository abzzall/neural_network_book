"""Verify the evidence contract and optionally compare a recorded reference run."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capstone_project import CHAPTER, run
sys.path.insert(0, str(CHAPTER))
from capstone_acceptance import validate


def check(path, reference):
    try:
        recorded = json.loads(Path(path).read_text())
        validate(recorded)
        for key in ('data_sha256', 'audit', 'split_ids', 'seeds', 'epochs',
                    'data_seed', 'split_seed', 'selection_order', 'selection_tolerance', 'chosen'):
            if recorded[key] != reference[key]:
                return False
        for key in ('representative_test_errors', 'low_signal_slice', 'representative_seed'):
            if recorded[key] != reference[key]:
                return False
        if not np.allclose(recorded['scaler_mean'], reference['scaler_mean'], rtol=0, atol=1e-12):
            return False
        for section in ('validation', 'test'):
            for name, expected in reference[section].items():
                if not np.allclose(recorded[section][name]['per_seed'], expected['per_seed'],
                                   rtol=0, atol=0.01):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


if __name__ == '__main__':
    reference = run()
    validate(reference)
    if len(sys.argv) > 1:
        ok = check(sys.argv[1], reference)
    else:
        ok = True
    print(f"selected: {reference['chosen']}")
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
