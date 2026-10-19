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


def test_boosting_scratch(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'boosting_scratch.py')], cwd=tmp_path, capture_output=True,
                           text=True, check=True, timeout=600).stdout.splitlines()
    assert lines[0] == ('best split x <= 3.5: leaf means 2.50 and 7.50, squared error 1.00 '
                        '(no split: 38.50)')
    assert float(lines[1].rsplit(' ', 1)[1]) < 1e-9          # equals scikit-learn
    assert lines[2:] == [
        '  1 trees: training MSE 23.590, validation MSE 21.252',
        ' 10 trees: training MSE 9.945, validation MSE 9.822',
        ' 50 trees: training MSE 1.620, validation MSE 3.076',
        '200 trees: training MSE 0.459, validation MSE 2.064']


def test_modern_boosting(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'modern_boosting.py')], cwd=tmp_path,
                           capture_output=True, text=True, check=True, timeout=600).stdout.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('unconstrained AUC ')
    assert lines[1].startswith('monotonic     AUC ')
    auc = [float(line.split()[2].rstrip(',')) for line in lines]
    assert min(auc) > 0.84
    assert lines[1].endswith('grid violations 0')
