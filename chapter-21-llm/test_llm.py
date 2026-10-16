"""Decoding rules, compute budget, DPO, and the GPT-2 demos (skipped without GPT-2)."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lm

GPT2 = (lm.directory('gpt2') / 'model.safetensors').is_file()


def run(script, cwd):
    return subprocess.run([sys.executable, str(HERE / script)], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=600).stdout.splitlines()


def test_decoding_rules(load_script):
    d = load_script('decoding')
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    np.testing.assert_allclose(d.top_k(p, 3)[:3], [0.5, 0.3125, 0.1875])
    np.testing.assert_allclose(d.top_p(p, 0.75), d.top_k(p, 3))
    np.testing.assert_allclose(d.top_p(p, 0.65), d.top_k(p, 2))   # exactly reaches p
    np.testing.assert_allclose(d.softmax(np.log(p), 0.5), p ** 2 / (p ** 2).sum())
    assert d.softmax(np.log(p), 1e-3).argmax() == 0


def test_decoding_script(tmp_path):
    assert run('decoding.py', tmp_path) == [
        'probabilities:     0.4000 0.2500 0.1500 0.1000 0.0600 0.0400',
        'temperature 0.5:   0.6149 0.2402 0.0865 0.0384 0.0138 0.0061',
        'temperature 2.0:   0.2773 0.2192 0.1698 0.1386 0.1074 0.0877',
        'top-k, k = 3:      0.5000 0.3125 0.1875 0.0000 0.0000 0.0000',
        'top-p, p = 0.75:   0.5000 0.3125 0.1875 0.0000 0.0000 0.0000',
        'top-p, p = 0.60:   0.6154 0.3846 0.0000 0.0000 0.0000 0.0000',
        'share of 10000 top-p draws per token: 0.4990 0.3122 0.1888 0.0000 0.0000 0.0000']


def test_compute_budget(load_script, tmp_path):
    c = load_script('compute_budget')
    for budget in (1e20, 5.04e23):
        n, d = c.optimal(budget)
        assert 6 * n * d == pytest.approx(budget)
        for scale in (0.8, 1.25):                  # the optimum beats nearby sizes
            assert c.loss(n, d) < c.loss(n * scale, budget / (6 * n * scale))
    lines = run('compute_budget.py', tmp_path)
    assert lines[2] == ('optimum for the Gopher budget: 30.3B parameters, 2.77T tokens, '
                        '91 tokens per parameter, loss 1.936')
    assert lines[3] == 'Chinchilla as trained: 20 tokens per parameter'


def test_dpo_recovers_the_kl_optimum(tmp_path):
    lines = run('dpo_toy.py', tmp_path)
    assert lines[0] == 'preference pairs: 15991'
    assert float(lines[4].split(': ')[1]) < 0.04
    implied = [float(v) for v in lines[5].split(': ')[1].split()]
    np.testing.assert_allclose(implied, [2, 1, 0, -1, -2], atol=0.1)


def test_missing_model_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        lm.load('gpt2', folder=tmp_path)


@pytest.mark.skipif(not GPT2, reason='run lm.py --download gpt2 first')
def test_perplexity_and_cache(tmp_path):
    assert run('perplexity.py', tmp_path)[:2] == [
        'original: 16 tokens, perplexity 48.0', 'shuffled: 16 tokens, perplexity 6051.7']
    lines = run('kv_cache_demo.py', tmp_path)
    assert lines[2] == 'identical tokens: True'
    assert lines[3] == 'cache for 65 tokens: 4.79 MB (12 layers, width 768)'
