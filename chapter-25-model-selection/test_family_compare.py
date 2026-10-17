"""The family comparison runs and gives the measured ordering (needs the credit data)."""

from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chapter-23-interpretability'))
import credit_data


@pytest.mark.skipif(not credit_data.PATH.is_file(),
                    reason='run chapter-23-interpretability/credit_data.py --download first')
def test_family_compare(tmp_path):
    out = subprocess.run([sys.executable, str(HERE / 'family_compare.py')], cwd=tmp_path,
                         capture_output=True, text=True, check=True, timeout=900).stdout
    scores = {}
    current = None
    for line in out.splitlines():
        if not line.startswith('  '):
            current = line.split(' (')[0]
        else:
            family = line[2:24].strip()
            scores[(current, family)] = float(line[24:].split()[0])
    assert scores[('credit', 'logistic regression')] == pytest.approx(0.7913, abs=0.01)
    assert scores[('credit', 'neural network (MLP)')] < scores[('credit', 'logistic regression')]
    for family in ('logistic regression', 'gradient boosting', 'neural network (MLP)'):
        assert scores[('breast cancer', family)] > 0.99
        assert 0.96 < scores[('digits', family)] < 0.98
