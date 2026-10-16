"""Reject missing and invalid recorded CUDA-exercise measurements."""

import json
from pathlib import Path
import subprocess
import sys


def test_recorded_results(tmp_path, load_script):
    verifier = load_script('verify_results')
    values = verifier.cuda_training.run()
    assert verifier.check(values)
    assert not verifier.check({})
    for key in values:
        bad = dict(values, **{key: float('nan')})
        assert not verifier.check(bad)
    bad = dict(values, cpu_final_mse=1.)
    assert not verifier.check(bad)
    record = tmp_path / 'results.json'
    record.write_text(json.dumps(values))
    script = Path(__file__).with_name('verify_results.py')
    result = subprocess.run([sys.executable, str(script), str(record)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == 'PASS'
    record.write_text(json.dumps(bad))
    result = subprocess.run([sys.executable, str(script), str(record)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and result.stdout.splitlines()[-1] == 'FAIL'
