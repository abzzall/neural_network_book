"""Sinusoidal positions and permutation equivariance of plain attention."""

import numpy as np

from attention_numpy import attention


def sinusoidal(length, width):
    positions = np.arange(length)[:, None]
    frequencies = 10000. ** (-np.arange(0, width, 2) / width)
    angles = positions * frequencies
    result = np.empty((length, width))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles[:, :width // 2])
    return result


if __name__ == '__main__':
    x = np.random.default_rng(42).normal(size=(3, 4))
    permutation = [2, 0, 1]
    plain, _ = attention(x, x, x)
    reordered, _ = attention(x[permutation], x[permutation], x[permutation])
    positions = sinusoidal(3, 4)
    positioned, _ = attention(x + positions, x + positions, x + positions)
    moved = x[permutation] + positions
    moved_output, _ = attention(moved, moved, moved)
    print(f'plain attention is permutation-equivariant: {np.allclose(reordered, plain[permutation])}')
    print(f'fixed positions break that equivalence: {not np.allclose(moved_output, positioned[permutation])}')
    print('position zero: ' + ', '.join(f'{v:.1f}' for v in positions[0]))
