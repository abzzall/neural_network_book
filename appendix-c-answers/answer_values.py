"""Appendix C: every number in the answers to selected review questions, computed."""

import importlib.util
from pathlib import Path

import numpy as np

from nnbook import Perceptron

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show(question, value):
    print(f'{question}: {value}')


show('2.1 parameters of a 784-128 layer', 784 * 128 + 128)
p = 0.95 * 0.001 / (0.95 * 0.001 + 0.02 * 0.999)
show('3.1 P(fraud | flagged)', f'{p:.4f}')
show('3.7 cross-entropy, true class at 0.3', f'{-np.log(0.3):.4f}')
show('4.10 class weights for 0.5 % positives',
     f'{1 / (2 * 0.995):.4f} and {1 / (2 * 0.005):.1f}')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
p_or = Perceptron().fit(X, np.array([-1, 1, 1, 1]))
show('5.2 perceptron on OR: mistakes per epoch', p_or.errors_per_epoch)
show('5.2 perceptron on OR: w and b', f'{p_or.w.tolist()} {p_or.b}')
show('6.1 parameters of 20-64-32-5', (20 + 1) * 64 + (64 + 1) * 32 + (32 + 1) * 5)
show('7.2 updates per epoch', 50_000 // 100)
show('12.1 sizes after conv and pool', f'{(32 + 2 * 2 - 5) // 1 + 1} then {32 // 2}')
show('12.2 convolution and dense parameters',
     f'{32 * (3 * 5 * 5 + 1)} and {32 * (64 * 64 * 3 + 1)}')
bpe = load('chapter-15-nlp/bpe.py')
merges, _ = bpe.learn({'rain': 5, 'train': 4, 'brain': 3, 'trains': 2, 'drain': 1}, 6)
show('15.3 "brains" after six merges', ' '.join(bpe.encode('brains', merges)))
layers = next(L for L in range(1, 20) if 1 + 2 * (2 ** L - 1) >= 365)
show('16.8 layers of kernel 3 to see a year', f'{layers} (receptive field {1 + 2 * (2 ** layers - 1)})')
show('17.2 intersection and union', f'{1.8 * 2:.1f} and {4 + 4 - 1.8 * 2:.1f}')
show('18.3 third GCN output', f'{2 / 3 + 3 / 3 + 2 / np.sqrt(6):.6f}')
show('19.1 returns', f'{1 + 0.5 + 0.25} and {1 + 0.9 + 0.81:.2f}')
probs = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
keep = int(np.searchsorted(np.cumsum(probs), 0.9 - 1e-12)) + 1
show('21.2 top-p set for p = 0.9', ' '.join(f'{v:.4f}' for v in probs[:keep] / probs[:keep].sum()))
show('21.4 GPT-2 cache for 1024 tokens (MB)', f'{2 * 12 * 1024 * 768 * 4 / 1e6:.1f}')
show('22.3 LoRA rank 4 on 768 x 768', f'{2 * 4 * 768} weights, {2 * 4 * 768 / 768 ** 2:.2%}')
quant = load('chapter-24-deployment/quantize_numpy.py')
x = np.array([-0.5, 0.0, 1.5])
q, s, z = quant.quantise(x)
error = np.abs(quant.dequantise(q, s, z) - x).max()
show('24.3 quantised codes', f'{q.tolist()}, scale {s:.6f}, zero point {z}, largest error {error:.5f}')
show('24.6 PSI', f'{0.3 * np.log(0.8 / 0.5) + (-0.3) * np.log(0.2 / 0.5):.4f}')
show('25.5 threshold for costs 2 and 3', 2 / (2 + 3))
show('25.6 cost of the policy', f'{(5 * 10 + 1 * (210 - 140)) / 300:.4f}')
