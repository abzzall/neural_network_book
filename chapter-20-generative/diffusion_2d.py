"""A denoising diffusion model on the eight-cluster ring (DDPM, small MLP)."""

import numpy as np
import torch
from torch import nn

from ring_data import mode_report, sample_ring

T = 100
BETAS = torch.linspace(1e-4, 0.2, T)                  # noise added at each step
ALPHAS = 1 - BETAS
ALPHA_BAR = torch.cumprod(ALPHAS, dim=0)


def noisy(x0, t, noise):
    """Jump straight to step t: x_t = sqrt(abar) x0 + sqrt(1 - abar) noise, eq. (20.9)."""
    abar = ALPHA_BAR[t][:, None]
    return abar.sqrt() * x0 + (1 - abar).sqrt() * noise


class Denoiser(nn.Module):
    """Predicts the noise in x_t; the step t enters as a scaled extra input."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T)[:, None]], dim=1))


def train(seed, steps=6000, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Denoiser()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x0 = torch.tensor(sample_ring(batch, rng), dtype=torch.float32)
        t = torch.randint(0, T, (batch,))
        noise = torch.randn_like(x0)
        loss = ((model(noisy(x0, t, noise), t) - noise) ** 2).mean()   # eq. (20.10)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


@torch.inference_mode()
def generate(model, n, seed=0):
    """Start from pure noise and remove the predicted noise step by step."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    for t in reversed(range(T)):
        eps = model(x, torch.full((n,), t))
        x = (x - BETAS[t] / (1 - ALPHA_BAR[t]).sqrt() * eps) / ALPHAS[t].sqrt()
        if t > 0:
            x = x + BETAS[t].sqrt() * torch.randn(n, 2, generator=g)
    return x.numpy()


if __name__ == '__main__':
    torch.set_num_threads(4)
    print(f'signal kept after {T} steps: sqrt(alpha_bar) = {ALPHA_BAR[-1].sqrt():.4f}')
    for seed in (0, 1, 2):
        modes, quality = mode_report(generate(train(seed), 5000))
        print(f'seed {seed}: modes covered {modes} of 8, samples near a mode {quality:.3f}')
