"""Check recorded sentiment-probe results when the optional weights are present."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pretrained_probe import CACHE


@pytest.mark.skipif(not (CACHE / 'model.safetensors').is_file(), reason='download pinned model first')
def test_probe_record(tmp_path, load_script):
    m = load_script('verify_results')
    record = tmp_path / 'results.json'
    values = dict(m.reference_values())
    record.write_text(json.dumps(values))
    assert m.check(record)
    values['test_accuracy_after'] = -1
    record.write_text(json.dumps(values))
    assert not m.check(record)
    script = Path(__file__).with_name('verify_results.py')
    result = subprocess.run([sys.executable, str(script), str(record)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 1 and result.stdout.splitlines()[-1] == 'FAIL'
