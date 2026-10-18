"""Rotary position embedding: the attention score depends only on the offset."""

import numpy as np


def rotate(x, position, base=10000.0):
    """Rotate each pair of components (2r, 2r+1) by the angle position / base^(2r/d)."""
    d = len(x)
    angles = position / base ** (np.arange(0, d, 2) / d)
    cos, sin = np.cos(angles), np.sin(angles)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * cos - x[1::2] * sin
    out[1::2] = x[0::2] * sin + x[1::2] * cos
    return out


rng = np.random.default_rng(0)
q, k = rng.normal(size=8), rng.normal(size=8)

if __name__ == '__main__':
    for m, n in ((3, 1), (12, 10), (50, 48), (5, 1)):
        score = rotate(q, m) @ rotate(k, n)
        print(f'query at {m:2d}, key at {n:2d} (offset {m - n}): score {score:.6f}')
    print(f'norm kept: {np.isclose(np.linalg.norm(rotate(q, 7)), np.linalg.norm(q))}')
