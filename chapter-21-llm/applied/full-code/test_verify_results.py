"""The exercise marker accepts measured results and rejects wrong or missing ones."""

from pathlib import Path
import subprocess
import sys


def test_rag_marker(load_script, tmp_path):
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
    values['correct_with_retrieval'] = 5
    write()
    assert not marker.check(path)
    values['correct_with_retrieval'] = marker.REFERENCE['correct_with_retrieval']
    values['questions'] = 19
    write()
    assert not marker.check(path)
    del values['questions']
    write()
    assert not marker.check(path)
    values['questions'] = 'not a number'
    write()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'MISMATCH questions' in result.stdout
