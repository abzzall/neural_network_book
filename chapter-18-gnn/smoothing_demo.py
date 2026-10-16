"""Repeated linear propagation on a regular cycle, without learned weights."""

import numpy as np

from gcn_numpy import normalise


def run():
    adjacency = np.zeros((8, 8))
    for index in range(8):
        adjacency[index, (index + 1) % 8] = 1
        adjacency[(index + 1) % 8, index] = 1
    operator = normalise(adjacency)
    hidden = np.eye(8)
    result = []
    for depth in range(33):
        if depth in (0, 1, 2, 4, 8, 16, 32):
            result.append((depth, float(np.mean((hidden - hidden.mean(0)) ** 2))))
        hidden = operator @ hidden
    return result


if __name__ == '__main__':
    for depth, variance in run():
        print(f'depth {depth:2d}: node variance {variance:.9f}')
