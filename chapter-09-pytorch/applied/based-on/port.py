"""Chapter 9 applied exercise (starting point): the dropout network of chapter 8, in PyTorch.

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
    raise NotImplementedError("Step 1: nn.Sequential(Linear(64, 256), ReLU, Dropout(0.5), Linear(256, 10))")


# Step 2: SGD with momentum, mini-batches of 32, 200 epochs, everything seeded.
def train(train_set, seed, epochs=200):
    raise NotImplementedError("Step 2: seed, DataLoader(batch 32, seeded generator), SGD(lr 0.05, momentum 0.9), 200 epochs")


# Step 3: accuracy in evaluation mode, without recording gradients.
def accuracy(model, data):
    raise NotImplementedError("Step 3: model.eval(), torch.inference_mode(), fraction of correct predictions")


# Step 4: validation accuracy over several seeds.
def seed_study(train_set, val, seeds=range(5)):
    return np.array([accuracy(train(train_set, s), val) for s in seeds])
