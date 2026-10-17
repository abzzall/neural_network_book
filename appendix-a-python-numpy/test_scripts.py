"""Run the appendix scripts as a reader would and check what they print."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, cwd):
    return subprocess.run([sys.executable, str(HERE / script)], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=300).stdout.splitlines()


def test_python_basics(tmp_path):
    assert run('python_basics.py', tmp_path) == [
        'list: [3.5, 2.0, 4.25, 1.25] length 4 last 1.25',
        "available: ['apples', 'plums']",
        'total 11.00 over 4 items',
        'mean: 2.75 mean of nothing: 0.0',
        "counts: {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}"]


def test_numpy_arrays(tmp_path):
    out = run('numpy_arrays.py', tmp_path)
    assert 'after changing the view, a[0, 0] = 100' in out
    assert 'after changing the copy, a[1, 0] = 4' in out
    assert 'even entries: [100   2   4   6   8  10]' in out
    assert out[-1] == 'float32 vector: [1. 2. 3.] float32  x @ x = 14.0'


def test_broadcasting(tmp_path):
    out = run('broadcasting.py', tmp_path)
    assert out[:4] == ['(3, 1) + (1, 4) -> (3, 4)', '[[ 1.  2.  3.  4.]',
                       ' [11. 12. 13. 14.]', ' [21. 22. 23. 24.]]']
    assert out[4] == 'same result: True'
    loop, vector = (float(v.split()[0]) for v in out[5].replace('loop ', '').split(', vectorised '))
    assert vector < loop


def test_plotting(tmp_path):
    assert run('plotting.py', tmp_path) == ['saved appendix_a_sigmoid.png with 2 curves']
    assert (HERE.parent / 'outputs' / 'appendix_a_sigmoid.png').stat().st_size > 1000
