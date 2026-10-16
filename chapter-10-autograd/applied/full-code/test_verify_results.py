"""Verify the marker accepts measured results and rejects missing or wrong values."""

from pathlib import Path
import subprocess
import sys


def test_marker(load_script, tmp_path):
    m = load_script('verify_results')
    values = m.reference_values()
    path = tmp_path / 'results.md'

    def write():
        path.write_text('\n'.join(f'| `{k}` | {v} |' for k, v in values.items()))

    write()
    assert m.check(path)
    script = Path(__file__).with_name('verify_results.py')
    result = subprocess.run([sys.executable, str(script), str(path)],
                            capture_output=True, text=True, check=True, timeout=60)
    assert all(line.startswith('OK ') for line in result.stdout.splitlines())
    values['final_mse'] = 0.5
    write()
    assert not m.check(path)
    del values['final_mse']
    write()
    assert not m.check(path)
    values['final_mse'] = 'not a number'
    write()
    result = subprocess.run([sys.executable, str(script), str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'MISMATCH final_mse' in result.stdout
