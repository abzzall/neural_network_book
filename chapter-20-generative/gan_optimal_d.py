"""The optimal discriminator for fixed data and generator distributions.

Real data N(1, 1), generated data N(-1, 1): eq. (20.7) gives D*(x) = sigmoid(2x).
A discriminator D(x) = sigmoid(w x + b) trained by binary cross-entropy should
learn w = 2 and b = 0; the value of the game is then -log 4 + 2 JSD.
"""

import math

import numpy as np
import torch

rng = np.random.default_rng(0)
real = torch.tensor(rng.normal(1.0, 1.0, 20000), dtype=torch.float32)
fake = torch.tensor(rng.normal(-1.0, 1.0, 20000), dtype=torch.float32)
w, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
optimiser = torch.optim.Adam([w, b], lr=0.05)
for _ in range(500):
    loss = -(torch.nn.functional.logsigmoid(w * real + b).mean()
             + torch.nn.functional.logsigmoid(-(w * fake + b)).mean())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
print(f'learned w = {w.item():.3f}, b = {b.item():.3f} (theory: 2, 0)')
value = -loss.item()                     # E log D(real) + E log(1 - D(fake))
print(f'value of the game: {value:.4f}')
print(f'implied Jensen-Shannon divergence: {(value + math.log(4)) / 2:.4f} nats')
x = np.linspace(-8, 8, 200001)
p, q = np.exp(-(x - 1) ** 2 / 2), np.exp(-(x + 1) ** 2 / 2)
p, q = p / p.sum(), q / q.sum()
m = (p + q) / 2
jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
print(f'Jensen-Shannon divergence by numerical integration: {jsd:.4f} nats')
