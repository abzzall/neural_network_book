"""Run the reference summary and check its reporting and rubric."""

import json
from pathlib import Path
import subprocess
import sys
import pytest


HERE = Path(__file__).resolve().parent


def test_demo(tmp_path):
    output = subprocess.run([sys.executable, str(HERE / 'project_demo.py')], cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=90).stdout.splitlines()
    assert output[0] == 'split rows: 720/240/240'
    assert len([line for line in output if 'validation macro-F1:' in line]) == 4
    assert any(line.startswith('chosen: ') for line in output)
    assert any(line.startswith('linear test macro-F1: ') for line in output)
    expected = {'linear validation macro-F1': .691934,
                'quadratic validation macro-F1': .928390,
                'mlp8 validation macro-F1': .893230,
                'mlp16 validation macro-F1': .893310,
                'linear test macro-F1': .682783,
                'quadratic test macro-F1': .895817}
    for line in output:
        if 'macro-F1:' in line:
            mean, std = map(float, line.split(': ')[1].split(' +/- '))
            assert 0 <= mean <= 1 and 0 <= std <= 1
            assert mean == pytest.approx(expected[line.split(': ')[0]], abs=0.005)
    assert 'chosen: quadratic' in output
    assert 'representative-seed test errors: 25' in output
    errors, rows = map(int, output[-1].split(': ')[1].split('/'))
    assert 0 <= errors <= rows <= 240


def test_rubric_total():
    rubric = json.loads((HERE / 'rubric.json').read_text())
    assert len(rubric) == 7 and sum(rubric.values()) == 100
    assert all(isinstance(points, int) and points > 0 for points in rubric.values())
