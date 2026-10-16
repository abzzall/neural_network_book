"""Run the chapter's scripts as a reader would and check what they print."""

from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def run(script, cwd):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=600)
    return result.stdout.splitlines()


def number(line, key):
    return float(line.split(key)[1].split()[0].rstrip(','))


def test_elbo_example(tmp_path):
    lines = run('elbo_example.py', tmp_path)
    assert lines[0].startswith('q = N(1.0, 0.64): expected log-likelihood -1.7389, '
                               'KL 0.5431, ELBO -2.2821, sampled -2.28')
    assert lines[1].startswith('q = N(1.0, 0.50): expected log-likelihood -1.6689, '
                               'KL 0.5966, ELBO -2.2655, sampled -2.26')
    assert lines[2] == 'log p(x) = -2.2655'


def test_optimal_discriminator(tmp_path):
    lines = run('gan_optimal_d.py', tmp_path)
    assert number(lines[0], 'w = ') == pytest.approx(2.0, abs=0.03)
    assert abs(number(lines[0], 'b = ')) < 0.03
    assert number(lines[2], 'divergence: ') == pytest.approx(0.3368, abs=0.003)
    assert lines[3] == 'Jensen-Shannon divergence by numerical integration: 0.3368 nats'


def test_anomaly_scan(tmp_path):
    lines = run('anomaly_scan.py', tmp_path)
    rows = [line.split() for line in lines[1:11]]
    pca = [float(r[1]) for r in rows]
    ae = [float(r[2].rstrip(',')) for r in rows]
    assert pca[1] == pytest.approx(0.690, abs=1e-3) and pca[2] == pytest.approx(0.894, abs=1e-3)
    assert lines[-1].startswith('mean over classes: PCA 0.880, autoencoder ')
    assert number(lines[-1], 'autoencoder ') == pytest.approx(0.912, abs=0.01)
    assert sum(a > p for a, p in zip(ae, pca)) >= 6


def test_vae_demo(tmp_path):
    lines = run('vae_demo.py', tmp_path)
    assert lines[0] == 'classifier accuracy on real test digits: 0.969'
    rows = [line.split() for line in lines[2:]]
    small = [r for r in rows if r[0] == '2']
    large = [r for r in rows if r[0] == '8']
    assert all(24 < float(r[2]) < 26 for r in rows)
    assert max(float(r[2]) for r in large) < min(float(r[2]) for r in small)
    assert min(int(r[5].strip('()')) for r in large) >= 50     # every digit generated


def test_gan_and_diffusion_cover_modes(tmp_path):
    gan = run('gan_2d.py', tmp_path)
    assert all(7 <= int(line.split('covered ')[1].split()[0]) <= 8 for line in gan)
    diffusion = run('diffusion_2d.py', tmp_path)
    assert diffusion[0] == 'signal kept after 100 steps: sqrt(alpha_bar) = 0.0046'
    assert all('covered 8 of 8' in line for line in diffusion[1:])
    assert all(number(line, 'mode ') > 0.8 for line in diffusion[1:])


def test_figure_data(tmp_path):
    out = subprocess.run([sys.executable, str(HERE / 'figure_data.py'), str(tmp_path)],
                         capture_output=True, text=True, check=True, timeout=600).stdout
    assert out.strip() == "modes covered: {'real': 8, 'gan': 7, 'diffusion': 8}"
    import numpy as np
    for name in ('real', 'gan', 'diffusion'):
        assert np.loadtxt(tmp_path / f'{name}.dat', skiprows=1).shape == (600, 2)
