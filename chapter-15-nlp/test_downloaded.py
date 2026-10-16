"""Scripts that need the SMS data or the pretrained encoder; skipped until downloaded."""

from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encoder
import spam_data

DATA = spam_data.PATH.is_file()
MODEL = (encoder.CACHE / 'model.safetensors').is_file()
needs_data = pytest.mark.skipif(not DATA, reason='run spam_data.py --download first')
needs_model = pytest.mark.skipif(not MODEL, reason='run encoder.py --download first')


def run(script, cwd):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=300)
    return result.stdout.splitlines()


def numbers(line):
    return [float(x) for x in line.replace(',', ' ').split() if x.replace('.', '').isdigit()]


def test_missing_files_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        spam_data.load(tmp_path / 'absent')
    with pytest.raises(FileNotFoundError, match='--download'):
        encoder.load_encoder(directory=tmp_path)


def test_split_removes_duplicates():
    texts = ['a', 'b', 'a', 'c', 'd', 'e', 'f', 'g'] * 2
    labels = [0, 1, 0, 0, 1, 0, 1, 0] * 2
    train, test, _, _ = spam_data.split(texts, labels, test_size=0.5)
    assert not set(train) & set(test) and len(train) + len(test) == 7


@needs_data
def test_data_summary(tmp_path):
    assert run('spam_data.py', tmp_path) == [
        'messages: 5574, spam: 747', 'distinct messages: 5171',
        'with duplicates: train 4180, test 1394, test also in train 161',
        'deduplicated: train 3878, test 1293, test also in train 0']


@needs_data
def test_tfidf_baseline(tmp_path):
    lines = run('tfidf_baseline.py', tmp_path)
    assert lines[0] == 'test messages: 1293 (spam 163)'
    tn, fp, fn, tp = numbers(lines[1])
    assert tn + fp == 1130 and fn + tp == 163 and fp <= 25 and fn <= 20
    assert float(lines[3].split(': ')[1]) == pytest.approx(0.9514, abs=0.01)
    assert {'call', 'txt', 'free'} <= set(lines[4].split(': ')[1].split(', '))
    assert lines[6] == 'messages containing &lt;: 222, of them spam: 0'


@needs_data
def test_duplicates_inflate_scores(tmp_path):
    lines = run('duplicates_demo.py', tmp_path)
    means = {line.split(': ')[0]: float(line.split(': ')[1].split(',')[0])
             for line in lines[1:]}
    assert means['random split, seen in training'] > means['random split, unseen']
    assert means['random split, all test'] > means['deduplicated split']
    assert means['deduplicated split'] == pytest.approx(0.9106, abs=0.01)


@needs_model
def test_encoder_and_search(tmp_path):
    tokenizer, model = encoder.load_encoder()
    vectors = encoder.embed(tokenizer, model, ['short', 'a longer sentence with padding'])
    assert vectors.shape == (2, 384)
    assert vectors.norm(dim=1).tolist() == pytest.approx([1, 1], abs=1e-5)
    lines = run('encoder.py', tmp_path)
    assert lines[0] == 'embedding size: 384'
    assert float(lines[1].split(': ')[1]) == pytest.approx(0.6758, abs=0.01)
    lines = run('semantic_search.py', tmp_path)
    assert lines[-1] == 'encoder correct: 4 of 4' and lines[-2] == 'tf-idf correct: 1 of 4'


@needs_model
def test_tokenizer_coverage(tmp_path):
    lines = run('tokenizer_coverage.py', tmp_path)
    assert lines[0] == 'English: 7 words -> 8 tokens, 0 unknown'
    assert lines[2] == 'Kazakh: 4 words -> 12 tokens, 3 unknown'
    assert lines[3].split()[:3] == ['[UNK]'] * 3
