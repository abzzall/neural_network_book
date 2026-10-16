"""Run the chapter's demonstrations and check the quantities they print."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, cwd, *args):
    result = subprocess.run([sys.executable, str(HERE / script), *args], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=120)
    return result.stdout.splitlines()


def test_normalise_demo(tmp_path):
    assert run('normalise_demo.py', tmp_path) == [
        'composed:   1 code points, 2 bytes: U+00E9',
        'decomposed: 2 code points, 3 bytes: U+0065 U+0301',
        'equal as strings: False',
        'equal after NFC: True',
        'NFC keeps compatibility forms: True',
        'NFKC result: file 2026',
        'Kazakh word: 9 code points, 18 bytes: U+049A U+0430 U+0437 U+0430 U+049B '
        'U+0441 U+0442 U+0430 U+043D',
        'first letter after casefold: 1 code points, 2 bytes: U+049B',
    ]


def test_bpe_demo(tmp_path):
    assert run('bpe.py', tmp_path) == [
        'merge 1: a + i (count 15)',
        'merge 2: ai + n (count 15)',
        'merge 3: r + ain (count 15)',
        'merge 4: rain + </w> (count 13)',
        'merge 5: t + rain</w> (count 4)',
        'merge 6: b + rain</w> (count 3)',
        'rains -> rain s </w>',
        'strain -> s train</w>',
        'grain -> g rain</w>',
    ]


def test_tfidf_demo(tmp_path):
    assert run('tfidf_numpy.py', tmp_path) == [
        'vocabulary: are call free me now prize to when win you',
        'idf: 1.6931 1.2877 1.0000 1.6931 1.2877 1.6931 1.6931 1.6931 1.2877 1.6931',
        'document 1: 0.0000 0.0000 0.3731 0.0000 0.4805 0.6317 0.0000 0.0000 0.4805 0.0000',
        'max difference from scikit-learn: 0.0e+00',
        'cosine(doc 1, doc 3): 0.5416',
    ]


def test_skipgram_demo(tmp_path):
    lines = run('skipgram_demo.py', tmp_path)
    losses = [float(line.split()[-1]) for line in lines[:5]]
    assert losses[-1] < losses[0]
    assert lines[5].startswith('nearest to cat: ')
    assert set(lines[5].split(': ')[1].split(', ')) == {'dog', 'horse', 'cow'}
    assert set(lines[6].split(': ')[1].split(', ')) == {'man', 'boy', 'girl'}
