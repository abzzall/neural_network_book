"""Chapter 9 applied exercise: the dropout network of chapter 8, in PyTorch.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn; the splits of chapter 8.
"""

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbook.preprocessing import StandardScaler, train_test_split


def load_splits():
    """The 200 training images, validation rows, and test rows of chapter 8, as tensors."""
    X, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
    small = np.random.default_rng(0).permutation(train)[:200]
    val = np.setdiff1d(train, small)
    scaler = StandardScaler().fit(X[small])
    return [(torch.tensor(scaler.transform(X[i]), dtype=torch.float32), torch.tensor(y[i]))
            for i in (small, val, test)]


# Step 1: Linear(64, 256), ReLU, Dropout(0.5), Linear(256, 10).
def build():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))


# Step 2: SGD with momentum, mini-batches of 32, 200 epochs, everything seeded.
def train(train_set, seed, epochs=200):
    torch.manual_seed(seed)                                # initial weights and dropout masks
    model = build()
    loader = DataLoader(TensorDataset(*train_set), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    return model


# Step 3: accuracy in evaluation mode, without recording gradients.
def accuracy(model, data):
    X, y = data
    model.eval()
    with torch.inference_mode():
        return (model(X).argmax(dim=1) == y).float().mean().item()


# Step 4: validation accuracy over several seeds.
def seed_study(train_set, val, seeds=range(5)):
    return np.array([accuracy(train(train_set, s), val) for s in seeds])
