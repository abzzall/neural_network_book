"""Decoding rules over a vector of next-token logits: temperature, top-k, top-p."""

import numpy as np


def softmax(logits, temperature=1.0):
    """Next-token probabilities, eq. (21.2); lower temperature sharpens them."""
    z = np.asarray(logits, float) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def top_k(probs, k):
    """Keep the k most probable tokens and renormalise."""
    keep = np.argsort(-probs)[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def top_p(probs, p):
    """Keep the smallest set of most probable tokens whose total reaches p, eq. (21.3)."""
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    size = int(np.searchsorted(cumulative, p - 1e-12) + 1)
    out = np.zeros_like(probs)
    out[order[:size]] = probs[order[:size]]
    return out / out.sum()


def sample(probs, rng):
    return int(rng.choice(len(probs), p=probs))


if __name__ == '__main__':
    probs = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    logits = np.log(probs)                         # any logits with these probabilities
    show = lambda q: ' '.join(f'{v:.4f}' for v in q)
    print('probabilities:    ', show(softmax(logits)))
    print('temperature 0.5:  ', show(softmax(logits, 0.5)))
    print('temperature 2.0:  ', show(softmax(logits, 2.0)))
    print('top-k, k = 3:     ', show(top_k(probs, 3)))
    print('top-p, p = 0.75:  ', show(top_p(probs, 0.75)))
    print('top-p, p = 0.60:  ', show(top_p(probs, 0.60)))
    rng = np.random.default_rng(0)
    draws = [sample(top_p(probs, 0.75), rng) for _ in range(10000)]
    print('share of 10000 top-p draws per token:', show(np.bincount(draws, minlength=6) / 1e4))
