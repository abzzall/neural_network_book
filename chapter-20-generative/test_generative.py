"""Unit checks: autoencoder scores, VAE terms, ring modes, diffusion forward process."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules


def test_autoencoder_scores_shape_and_order(load_script):
    m = load_script('autoencoder')
    rng = np.random.default_rng(0)
    images = rng.random((200, 64)) * 0.1                     # dim, similar images
    model = m.train(images, epochs=30)
    scores = m.reconstruction_error(model, np.vstack([images[:5], np.ones((5, 64))]))
    assert scores.shape == (10,) and scores[5:].min() > scores[:5].max()


def test_vae_kl_matches_sampling(load_script):
    m = load_script('vae')
    torch.manual_seed(0)
    model = m.VAE(latent=2)
    x = torch.rand(4, 64)
    h = model.encoder(x)
    mu, log_var = model.mean(h), model.log_var(h)
    closed = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(dim=1)
    q = torch.distributions.Normal(mu, (0.5 * log_var).exp())
    z = q.sample((200_000,))
    p = torch.distributions.Normal(0.0, 1.0)
    sampled = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean(0)
    torch.testing.assert_close(closed, sampled, atol=2e-2, rtol=2e-2)


def test_mode_report(load_script):
    m = load_script('ring_data')
    rng = np.random.default_rng(0)
    assert m.mode_report(m.sample_ring(4000, rng)) == (8, 1.0)
    two_modes = np.repeat(m.CENTRES[:2], 500, axis=0)
    assert m.mode_report(two_modes)[0] == 2
    assert m.mode_report(np.zeros((100, 2))) == (0, 0.0)


def test_diffusion_closed_form_matches_step_by_step(load_script):
    m = load_script('diffusion_2d')
    torch.manual_seed(0)
    x0 = torch.full((200_000, 2), 1.5)
    x = x0.clone()
    for t in range(60):                                       # noise 60 steps one by one
        x = m.ALPHAS[t].sqrt() * x + m.BETAS[t].sqrt() * torch.randn_like(x)
    jumped = m.noisy(x0, torch.full((len(x0),), 59), torch.randn_like(x0))
    abar = m.ALPHA_BAR[59]
    for sample in (x, jumped):
        assert sample.mean().item() == pytest.approx(1.5 * abar.sqrt().item(), abs=0.01)
        assert sample.var().item() == pytest.approx(1 - abar.item(), abs=0.01)
    assert m.ALPHA_BAR[-1].sqrt() < 0.01                      # x_T is almost pure noise
