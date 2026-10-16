"""A GAN on the eight-cluster ring: how many modes does the generator cover?"""

import numpy as np
import torch
from torch import nn

from ring_data import mode_report, sample_ring


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, n_out))


def train(seed, steps=3000, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator, discriminator = mlp(2, 2), mlp(2, 1)               # D outputs a logit
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.functional.binary_cross_entropy_with_logits
    ones, zeros = torch.ones(batch, 1), torch.zeros(batch, 1)
    for _ in range(steps):
        real = torch.tensor(sample_ring(batch, rng), dtype=torch.float32)
        fake = generator(torch.randn(batch, 2))
        # discriminator step: real -> 1, generated -> 0, eq. (20.6)
        loss_d = bce(discriminator(real), ones) + bce(discriminator(fake.detach()), zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # generator step, non-saturating form: make D call the fakes real
        loss_g = bce(discriminator(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
    with torch.inference_mode():
        return generator(torch.randn(5000, 2)).numpy()


if __name__ == '__main__':
    torch.set_num_threads(4)
    for seed in (0, 1, 2):
        modes, quality = mode_report(train(seed))
        print(f'seed {seed}: modes covered {modes} of 8, '
              f'samples near a mode {quality:.3f}')
