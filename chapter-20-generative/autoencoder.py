"""A small autoencoder for 8x8 digit images and its reconstruction-error score."""

import numpy as np
import torch
from torch import nn


def make_autoencoder(code_size=8):
    encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, code_size), nn.ReLU())
    decoder = nn.Sequential(nn.Linear(code_size, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    return nn.Sequential(encoder, decoder)


def train(images, seed=0, epochs=150, batch=64, lr=1e-3):
    """Minimise the mean squared reconstruction error, eq. (20.1)."""
    torch.manual_seed(seed)
    model = make_autoencoder()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(images, dtype=torch.float32)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=order).split(batch):
            loss = ((model(x[idx]) - x[idx]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model.eval()


def reconstruction_error(model, images):
    """Anomaly score: mean squared error per image (high = unlike the training data)."""
    x = torch.tensor(images, dtype=torch.float32)
    with torch.inference_mode():
        return ((model(x) - x) ** 2).mean(dim=1).numpy()


def pca_error(pca, images):
    """The same score for a PCA model: a linear autoencoder."""
    return np.mean((pca.inverse_transform(pca.transform(images)) - images) ** 2, axis=1)
