"""The diagnostic records model failures rather than requiring success."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lm import directory


def test_controlled_evidence_contract(load_script):
    m = load_script('rag_failures')
    assert [row[0] for row in m.CASES] == ['supported', 'missing', 'contradiction']
    assert m.CASES[0][-1] in m.CASES[0][2]
    assert 'code' not in m.CASES[1][2]
    assert 'CEDAR-47' in m.CASES[2][2] and 'MAPLE-82' in m.CASES[2][2]
    assert 'INSUFFICIENT EVIDENCE' in m.messages('Q', 'E')[0]['content']


@pytest.mark.skipif(not (directory('qwen') / 'model.safetensors').is_file(),
                    reason='run lm.py --download qwen first')
def test_failure_diagnostic_runs_and_preserves_answers(tmp_path):
    result = subprocess.run([sys.executable, str(HERE / 'rag_failures.py')],
                            cwd=tmp_path, text=True, capture_output=True, check=True, timeout=300)
    records = json.loads(result.stdout)
    assert len(records) == 3
    for record in records:
        assert record['answer'].strip()
        assert record['exact_match'] == (record['answer'].strip().casefold()
                                         == record['expected'].casefold())
