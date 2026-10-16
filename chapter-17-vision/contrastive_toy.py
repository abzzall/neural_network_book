"""A symmetric SimCLR-style objective on original synthetic vectors."""

import torch
from torch import nn
from torch.nn import functional as F


def info_nce(first, second, temperature=.2):
    if first.shape != second.shape or len(first) < 2 or temperature <= 0:
        raise ValueError('matching views, at least two pairs and positive temperature required')
    z = F.normalize(torch.cat([first, second]), dim=-1)
    scores = z @ z.T / temperature
    scores = scores.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf)
    positive = (torch.arange(len(z), device=z.device) + len(first)) % len(z)
    return F.cross_entropy(scores, positive)


def run(steps=200):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(64, 2, generator=generator)
    encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8))
    projection = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    parameters = list(encoder.parameters()) + list(projection.parameters())
    optimiser = torch.optim.Adam(parameters, lr=.01)
    first = x + .05 * torch.randn(x.shape, generator=generator)
    second = x + .05 * torch.randn(x.shape, generator=generator)
    diagnostic = (first.clone(), second.clone())
    initial = info_nce(projection(encoder(first)), projection(encoder(second))).item()
    for _ in range(steps):
        first = x + .05 * torch.randn(x.shape, generator=generator)
        second = x + .05 * torch.randn(x.shape, generator=generator)
        optimiser.zero_grad(set_to_none=True)
        loss = info_nce(projection(encoder(first)), projection(encoder(second)))
        loss.backward()
        optimiser.step()
    with torch.inference_mode():
        first, second = diagnostic
        final = info_nce(projection(encoder(first)), projection(encoder(second))).item()
    return initial, final


if __name__ == '__main__':
    initial, final = run()
    print(f'initial contrastive loss: {initial:.6f}')
    print(f'final contrastive loss: {final:.6f}')
