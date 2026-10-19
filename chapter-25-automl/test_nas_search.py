from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def test_nas_search(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'nas_search.py')], cwd=tmp_path,
                           capture_output=True, text=True, check=True, timeout=600).stdout.splitlines()
    assert lines[0].startswith('candidates 8, selected ')
    assert lines[1].startswith('validation accuracy ')
    assert float(lines[1].split()[-1]) > 0.94
    assert lines[2].startswith('search seconds ')
