"""The exercise marker accepts measured results and rejects wrong or missing ones."""

from pathlib import Path
import subprocess
import sys


def test_forecast_marker(load_script, tmp_path):
    marker = load_script('verify_results')
    values = dict(marker.reference_values())
    path = tmp_path / 'results.md'

    def write():
        path.write_text('\n'.join(f'| `{k}` | {v} |' for k, v in values.items()))

    write()
    assert marker.check(path)
    command = [sys.executable, str(Path(__file__).with_name('verify_results.py')), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    assert all(line.startswith('OK ') for line in result.stdout.splitlines())
    values['test_gru_mae'] = 100.
    write()
    assert not marker.check(path)
    del values['test_gru_mae']
    write()
    assert not marker.check(path)
    values['test_gru_mae'] = 'not a number'
    write()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'MISMATCH test_gru_mae' in result.stdout
