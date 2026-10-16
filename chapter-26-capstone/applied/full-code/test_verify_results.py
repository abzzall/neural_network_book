"""Reject inconsistent records and records from a different experiment."""

import copy
import json
from pathlib import Path
import subprocess
import sys


def test_recorded_results(tmp_path, load_script):
    verifier = load_script('verify_results')
    reference = verifier.run()
    record = tmp_path / 'results.json'
    record.write_text(json.dumps(reference))
    assert verifier.check(record, reference)
    bad = copy.deepcopy(reference)
    bad['data_sha256'] = '0' * 64
    record.write_text(json.dumps(bad))
    assert not verifier.check(record, reference)
    bad = copy.deepcopy(reference)
    bad['validation']['linear']['per_seed'] = [.1] * 3
    bad['validation']['linear']['mean'] = .1
    bad['validation']['linear']['std'] = 0.
    record.write_text(json.dumps(bad))
    assert not verifier.check(record, reference)
    script = Path(__file__).with_name('verify_results.py')
    result = subprocess.run([sys.executable, str(script), str(record)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 1 and result.stdout.splitlines()[-1] == 'FAIL'
