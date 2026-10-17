"""Saliency and integrated gradients for a digit classifier, with two checks."""

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


def saliency(model, x, target):
    """Absolute gradient of the target logit with respect to each pixel."""
    x = x.clone().requires_grad_(True)
    model(x[None])[0, target].backward()
    return x.grad.abs()


def integrated_gradients(model, x, target, baseline=None, steps=64):
    """(x - x') times the average gradient on the straight path from x' to x, eq. (23.3)."""
    baseline = torch.zeros_like(x) if baseline is None else baseline
    alphas = (torch.arange(steps) + 0.5) / steps                  # midpoint rule
    path = (baseline + alphas[:, None] * (x - baseline)).requires_grad_(True)
    model(path)[:, target].sum().backward()
    return (x - baseline) * path.grad.mean(dim=0)


def train(x, y, seed=0, epochs=300):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


if __name__ == '__main__':
    images, labels = load_digits(return_X_y=True)
    tr, te, y_tr, y_te = train_test_split(images / 16, labels, test_size=0.3, random_state=0)
    x_tr, x_te = torch.tensor(tr, dtype=torch.float32), torch.tensor(te, dtype=torch.float32)
    model = train(x_tr, torch.tensor(y_tr))
    print(f'test accuracy: {(model(x_te).argmax(1).numpy() == y_te).mean():.4f}')
    errors, same_sal, same_ig = [], [], []
    torch.manual_seed(1)
    random_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    for x, t in zip(x_te[:100], y_te[:100]):
        ig = integrated_gradients(model, x, int(t))
        with torch.no_grad():
            gap = model(x[None])[0, t] - model(torch.zeros(1, 64))[0, t]
        errors.append(abs(ig.sum().item() - gap.item()))           # completeness
        same_sal.append(spearmanr(saliency(model, x, int(t)), saliency(random_model, x, int(t)))[0])
        same_ig.append(spearmanr(ig, integrated_gradients(random_model, x, int(t)))[0])
    print(f'completeness: largest |sum of attributions - logit gap| = {max(errors):.4f}')
    x, t = x_te[0], int(y_te[0])
    with torch.no_grad():
        gap = (model(x[None])[0, t] - model(torch.zeros(1, 64))[0, t]).item()
    for steps in (4, 16, 64, 256):
        error = abs(integrated_gradients(model, x, t, steps=steps).sum().item() - gap)
        print(f'  first test image, {steps:3d} steps: error {error:.5f}')
    print(f'rank correlation with a random network: saliency {np.nanmean(same_sal):.3f}, '
          f'integrated gradients {np.nanmean(same_ig):.3f}')
