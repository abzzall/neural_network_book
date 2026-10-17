"""The exercise marker accepts measured results and rejects wrong or missing ones."""

from pathlib import Path
import subprocess
import sys


def test_decision_marker(load_script, tmp_path):
    marker = load_script('verify_results')
    values = dict(marker.REFERENCE)
    path = tmp_path / 'results.md'

    def write():
        path.write_text('\n'.join(f'| `{k}` | {v} |' for k, v in values.items()))

    write()
    assert marker.check(path)
    command = [sys.executable, str(Path(__file__).with_name('verify_results.py')), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    assert all(line.startswith('OK ') for line in result.stdout.splitlines())
    values['logistic_at_cost_threshold_cost'] = 0.9
    write()
    assert not marker.check(path)
    values['logistic_at_cost_threshold_cost'] = marker.REFERENCE['logistic_at_cost_threshold_cost']
    values['approve_all_cost'] = 1.4
    write()
    assert not marker.check(path)
    del values['approve_all_cost']
    write()
    assert not marker.check(path)
    values['approve_all_cost'] = 'not a number'
    write()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'MISMATCH approve_all_cost' in result.stdout
