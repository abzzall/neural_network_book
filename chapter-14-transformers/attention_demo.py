"""Inspect every matrix in a three-position attention calculation."""

import numpy as np

from attention_numpy import attention


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    k = q.copy()
    v = np.array([[1., 0.], [0., 2.], [3., 1.]])
    out, cache = attention(q, k, v)
    causal, masked_cache = attention(q, k, v, np.tri(3, dtype=bool))
    for name, matrix in [('Q', q), ('K', k), ('V', v),
                         ('scaled scores', q @ k.T / np.sqrt(2)),
                         ('weights', cache[-1]), ('output', out),
                         ('causal weights', masked_cache[-1]),
                         ('causal output', causal)]:
        print(f'{name}:\n{matrix}')
