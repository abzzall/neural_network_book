"""Policy evaluation and value iteration on the grid world (Bellman equations)."""

from gridworld import ACTIONS, START, STATES, TERMINAL, show, step

GAMMA = 0.9


def backup(V, s, a):
    nxt, reward, done = step(s, a)
    return reward + (0.0 if done else GAMMA * V[nxt])


def evaluate_random_policy(tol=1e-10):
    """Iterate the Bellman expectation equation (19.3) for the uniform policy."""
    V = {s: 0.0 for s in STATES}
    while True:
        new = {s: 0.0 if s in TERMINAL else
               sum(backup(V, s, a) for a in ACTIONS) / len(ACTIONS) for s in STATES}
        if max(abs(new[s] - V[s]) for s in STATES) < tol:
            return new
        V = new


def value_iteration(sweeps=None, tol=1e-10):
    """Synchronous sweeps of the Bellman optimality equation (19.4)."""
    V, n = {s: 0.0 for s in STATES}, 0
    while sweeps is None or n < sweeps:
        new = {s: 0.0 if s in TERMINAL else
               max(backup(V, s, a) for a in ACTIONS) for s in STATES}
        n += 1
        done = max(abs(new[s] - V[s]) for s in STATES) < tol
        V = new
        if done and sweeps is None:
            break
    return V, n


def greedy_policy(V):
    return {s: max(ACTIONS, key=lambda a: backup(V, s, a))
            for s in STATES if s not in TERMINAL}


if __name__ == '__main__':
    for k in (1, 2):
        print(f'after sweep {k}:')
        show(value_iteration(sweeps=k)[0])
    V, n = value_iteration()
    print(f'converged after {n} sweeps:')
    show(V)
    policy = greedy_policy(V)
    print('greedy action from the start:', policy[START])
    print(f'random policy, value of the start: {evaluate_random_policy()[START]:.4f}')
