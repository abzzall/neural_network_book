"""Shared helpers: evaluating a fixed policy and detecting when a task is solved."""

import numpy as np

from cartpole import CartPole

SOLVED = 475.0           # Gymnasium's reward threshold for CartPole-v1


def evaluate(act, seed, episodes=20):
    """Mean return of a deterministic policy on fresh episodes."""
    env, returns = CartPole(seed=1000 + seed), []
    for _ in range(episodes):
        state, done, total = env.reset(), False, 0.0
        while not done:
            state, reward, terminated, truncated = env.step(act(state))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def solved_at(returns, window=100, threshold=SOLVED):
    """First episode (1-based) at which the mean of the last `window` returns
    reaches the threshold, or None."""
    returns = np.asarray(returns, float)
    for i in range(window, len(returns) + 1):
        if returns[i - window:i].mean() >= threshold:
            return i
    return None
