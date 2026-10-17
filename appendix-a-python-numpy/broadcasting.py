"""Broadcasting and vectorisation: the same result without Python loops."""

import time

import numpy as np

col = np.array([[0.0], [10.0], [20.0]])         # shape (3, 1)
row = np.array([[1.0, 2.0, 3.0, 4.0]])          # shape (1, 4)
print('(3, 1) + (1, 4) ->', (col + row).shape)
print(col + row)

rng = np.random.default_rng(0)
points = rng.normal(size=(500, 2))


def distances_loop(p):
    n = len(p)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.sqrt(((p[i] - p[j]) ** 2).sum())
    return out


def distances_vectorised(p):
    diff = p[:, None, :] - p[None, :, :]        # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    return np.sqrt((diff ** 2).sum(axis=2))


start = time.perf_counter()
slow = distances_loop(points)
loop_time = time.perf_counter() - start
start = time.perf_counter()
fast = distances_vectorised(points)
vector_time = time.perf_counter() - start
print('same result:', np.allclose(slow, fast))
print(f'loop {loop_time:.3f} s, vectorised {vector_time:.4f} s')
