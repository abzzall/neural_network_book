"""Chapter 4: an imbalanced dataset, stratified splitting, and class weights."""

import numpy as np

from nnbook.preprocessing import train_test_split

rng = np.random.default_rng(3)
y = (rng.random(2000) < 0.01).astype(int)          # about 1 fraud in 100

always_no = np.mean(y == 0)                         # accuracy of always saying "no fraud"
plain = [y[train_test_split(len(y), 0.1, seed=s)[1]].sum() for s in range(200)]
strat = [y[train_test_split(len(y), 0.1, seed=s, stratify=y)[1]].sum() for s in range(200)]
counts = np.bincount(y)
weights = len(y) / (2 * counts)                     # n / (K * n_k)

if __name__ == "__main__":
    print(f"positives: {counts[1]} of {len(y)}")
    print(f"accuracy of always predicting the majority: {always_no:.4f}")
    print(f"positives in 200 plain 10% test splits: {min(plain)} to {max(plain)}")
    print(f"positives in 200 stratified 10% test splits: {min(strat)} to {max(strat)}")
    print("class weights:", np.round(weights, 3))
