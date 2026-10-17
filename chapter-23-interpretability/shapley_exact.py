"""Exact Shapley values by enumerating every coalition of features."""

from itertools import combinations
from math import factorial

import numpy as np


def coalition_value(f, x, baseline, members):
    """f with the features in `members` taken from x and the rest from the baseline."""
    z = np.array(baseline, float)
    z[list(members)] = np.asarray(x, float)[list(members)]
    return f(z)


def shapley(f, x, baseline):
    """phi_i = sum over S without i of |S|!(n-|S|-1)!/n! [v(S + i) - v(S)], eq. (23.2)."""
    n = len(x)
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(others, size):
                gain = (coalition_value(f, x, baseline, S + (i,))
                        - coalition_value(f, x, baseline, S))
                phi[i] += weight * gain
    return phi


if __name__ == '__main__':
    f = lambda z: 2 * z[0] + z[1] * z[2]
    x, baseline = np.array([1.0, 2.0, 3.0]), np.zeros(3)
    phi = shapley(f, x, baseline)
    print('Shapley values:', ' '.join(f'{v:.4f}' for v in phi))
    print(f'sum {phi.sum():.4f} = f(x) - f(baseline) = {f(x) - f(baseline):.4f}')
    for S in [(), (0,), (1,), (2,), (1, 2), (0, 1, 2)]:
        print(f'v({set(S) or "{}"}) = {coalition_value(f, x, baseline, S):.1f}')
