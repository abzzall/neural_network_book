"""Chapter 3: entropy, cross-entropy, and the asymmetry of the KL divergence."""

import numpy as np


def entropy(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log2(p))


def cross_entropy(p, q):
    return -np.sum(np.asarray(p) * np.log2(q))


def kl(p, q):
    return cross_entropy(p, q) - entropy(p)


p = np.array([0.7, 0.2, 0.1])   # the true distribution
q = np.array([0.4, 0.4, 0.2])   # a model of it

if __name__ == "__main__":
    print(f"H(fair coin) = {entropy([0.5, 0.5]):.4f} bits")
    print(f"H(coin with p = 0.9) = {entropy([0.9, 0.1]):.4f} bits")
    print(f"H(p) = {entropy(p):.4f}, H(p, q) = {cross_entropy(p, q):.4f} bits")
    print(f"KL(p || q) = {kl(p, q):.4f}, KL(q || p) = {kl(q, p):.4f} bits")
