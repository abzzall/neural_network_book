"""Chapter 3: maximum likelihood for a coin, by formula and by gradient descent."""

import numpy as np

tosses = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])   # 1 = heads; 7 heads in 10
k, n = tosses.sum(), tosses.size


def bce(p):
    """Average negative log-likelihood (binary cross-entropy), in nats."""
    return -(k * np.log(p) + (n - k) * np.log(1 - p)) / n


def bce_gradient(p):
    return -(k / p - (n - k) / (1 - p)) / n


p_formula = k / n                                     # closed-form maximiser

p = 0.5                                               # gradient descent from a fair coin
for _ in range(200):
    p -= 0.1 * bce_gradient(p)

if __name__ == "__main__":
    print(f"heads = {k} of {n}")
    print(f"maximum-likelihood estimate k/n = {p_formula:.4f}")
    print(f"gradient descent on the cross-entropy: p = {p:.4f}")
    print(f"cross-entropy at p = 0.5: {bce(0.5):.4f}, at p = 0.7: {bce(0.7):.4f}")
