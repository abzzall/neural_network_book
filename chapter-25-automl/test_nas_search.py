from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def test_nas_search(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'nas_search.py')], cwd=tmp_path,
                           capture_output=True, text=True, check=True, timeout=600).stdout.splitlines()
    assert len(lines) == 4
    assert lines[0] == 'candidates 8, selected (64,), alpha 0.0001'
    assert lines[1].startswith('validation accuracy ')
    validation, test = float(lines[1].split()[2].rstrip(',')), float(lines[1].split()[-1])
    assert validation == pytest.approx(0.9472, abs=0.005)
    assert test == pytest.approx(0.9639, abs=0.005)
    assert lines[2].startswith('search seconds ')
    assert lines[3].startswith('final refit seconds ')
    assert all(float(line.split()[-1]) > 0 for line in lines[2:])
