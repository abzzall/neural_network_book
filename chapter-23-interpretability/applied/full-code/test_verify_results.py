"""The exercise marker accepts measured results and rejects wrong or missing ones."""

from pathlib import Path
import subprocess
import sys


def test_fairness_marker(load_script, tmp_path):
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
    values['group_threshold_odds_gap'] = 0.5
    write()
    assert not marker.check(path)
    values['group_threshold_odds_gap'] = marker.REFERENCE['group_threshold_odds_gap']
    values['with_age_accuracy'] = 0.9
    write()
    assert not marker.check(path)
    del values['with_age_accuracy']
    write()
    assert not marker.check(path)
    values['with_age_accuracy'] = 'not a number'
    write()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'MISMATCH with_age_accuracy' in result.stdout
