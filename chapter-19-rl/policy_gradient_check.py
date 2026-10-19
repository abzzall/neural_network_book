"""Enumerate a two-step MDP to check the discounted policy gradient exactly."""

from itertools import product

import numpy as np


def probability(theta):
    """Both states use the same Bernoulli policy pi(1|s) = sigmoid(theta)."""
    return np.exp(-np.logaddexp(0, -theta))


def objective(theta, gamma=0.5):
    """r1=0 and r2=a1, so E[G0] = gamma * pi(1|s1)."""
    return gamma * probability(theta)


def expected_estimate(theta, gamma=0.5, outer_discount=True, baseline=(0.0, 0.0)):
    """Sum the score-function estimate over all four possible trajectories."""
    p, expectation = probability(theta), 0.0
    for a0, a1 in product((0, 1), repeat=2):
        mass = (p if a0 else 1 - p) * (p if a1 else 1 - p)
        returns = (gamma * a1, float(a1))
        score = 0.0
        for t, action in enumerate((a0, a1)):
            weight = gamma ** t if outer_discount else 1.0
            score += weight * (returns[t] - baseline[t]) * (action - p)
        expectation += mass * score
    return expectation


if __name__ == '__main__':
    theta, h = 0.0, 1e-5
    finite_difference = (objective(theta + h) - objective(theta - h)) / (2 * h)
    print(f'objective: {objective(theta):.6f}')
    print(f'finite-difference gradient: {finite_difference:.6f}')
    print(f'expected discounted estimate: {expected_estimate(theta):.6f}')
    print(f'without outer discount: {expected_estimate(theta, outer_discount=False):.6f}')
