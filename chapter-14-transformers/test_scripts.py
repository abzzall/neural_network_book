"""Run each small CPU example from outside the repository."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def run(name, cwd):
    return subprocess.run([sys.executable, str(HERE / f'{name}.py')], cwd=cwd,
        capture_output=True, text=True, check=True, timeout=60).stdout.splitlines()


def test_attention_demo(tmp_path):
    lines = run('attention_demo', tmp_path)
    assert lines[:4] == ['Q:', '[[1. 0.]', ' [0. 1.]', ' [1. 1.]]']
    assert lines[20:24] == ['output:', '[[1.604448 0.796664]',
                           ' [1.401112 1.203336]', ' [1.758725 1.      ]]']
    assert lines[-4:] == ['causal output:', '[[1.       0.      ]',
                          ' [0.330238 1.339523]', ' [1.758725 1.      ]]']


def test_position_demo(tmp_path):
    assert run('position_demo', tmp_path) == [
        'plain attention is permutation-equivariant: True',
        'fixed positions break that equivalence: True',
        'position zero: 0.0, 1.0, 0.0, 1.0']


def test_character_demo(tmp_path):
    lines = run('character_demo', tmp_path)
    initial, final = [float(line.split(': ')[1]) for line in lines[:2]]
    assert 0 < final < initial / 10
    sample = lines[2].split(': ')[1]
    assert len(sample) == 48 and sample.startswith('rain falls.')
    assert set(sample) <= set('rain falls. sun shines. wind blows. ')
