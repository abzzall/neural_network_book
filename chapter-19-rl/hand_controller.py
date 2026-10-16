"""A one-line controller written from physical intuition, scored like a learned policy."""

from rl_utils import evaluate


def push_towards_fall(state):
    """Push the cart in the direction the pole is falling (angle plus half its rate)."""
    x, x_dot, theta, theta_dot = state
    return int(theta + 0.5 * theta_dot > 0)


if __name__ == '__main__':
    for seed in (0, 1, 2):
        print(f'seed {seed}: mean return over 20 episodes = '
              f'{evaluate(push_towards_fall, seed):.1f}')
