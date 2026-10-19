"""Run the completed reference project and print the report's main evidence."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'applied' / 'full-code'))
from capstone_project import run


if __name__ == '__main__':
    result = run()
    print('split rows: ' + '/'.join(str(p['rows']) for p in result['audit']['partitions'].values()))
    for name, values in result['validation'].items():
        print(f"{name} validation macro-F1: {values['mean']:.6f} +/- {values['std']:.6f}")
    print(f"chosen: {result['chosen']}")
    for name, values in result['test'].items():
        print(f"{name} test macro-F1: {values['mean']:.6f} +/- {values['std']:.6f}")
    print(f"representative-seed test errors: {len(result['representative_test_errors'])}")
    diagnostic = result['low_signal_slice']
    print(f"low-signal errors/rows: {diagnostic['errors']}/{diagnostic['rows']}")
