"""Tabular Q-learning on the grid world, compared with value iteration."""

import numpy as np

from gridworld import ACTIONS, START, STATES, TERMINAL, step
from value_iteration import GAMMA, value_iteration

NAMES = list(ACTIONS)


def q_learning(episodes=500, alpha=0.5, epsilon=0.1, seed=0, max_steps=100):
    """Learn Q from experience only; the agent never reads the transition rule."""
    rng = np.random.default_rng(seed)
    Q = {s: np.zeros(len(NAMES)) for s in STATES}
    visits = {s: 0 for s in STATES}
    lengths = []
    for _ in range(episodes):
        s, t = START, 0
        while s not in TERMINAL and t < max_steps:
            if rng.random() < epsilon:                      # explore
                a = int(rng.integers(len(NAMES)))
            else:                                           # exploit, ties at random
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            visits[s] += 1
            nxt, reward, done = step(s, NAMES[a])
            target = reward + (0.0 if done else GAMMA * Q[nxt].max())
            Q[s][a] += alpha * (target - Q[s][a])           # update, eq. (19.5)
            s, t = nxt, t + 1
        lengths.append(t)
    return Q, lengths, visits


if __name__ == '__main__':
    V_star, _ = value_iteration()
    for seed in (0, 1, 2):
        Q, lengths, visits = q_learning(seed=seed)
        start_error = abs(Q[START].max() - V_star[START])
        wrong = [s for s in STATES if s not in TERMINAL
                 and abs(Q[s].max() - V_star[s]) > 0.01]
        print(f'seed {seed}: error at start {start_error:.4f}, '
              f'steps (last 50 episodes) {np.mean(lengths[-50:]):.2f}, '
              f'states off by > 0.01: {[(s, visits[s]) for s in wrong]}')
