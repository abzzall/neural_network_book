"""The cart-pole balancing task, written out: the same physics as Gymnasium CartPole-v1.

State: cart position x, cart velocity, pole angle theta (radians), angular velocity.
Actions: 0 pushes the cart left, 1 pushes it right. Reward +1 per step; the episode
ends when |x| > 2.4 or |theta| > 12 degrees, or is cut off after 500 steps.
"""

import math

import numpy as np

GRAVITY, CART_MASS, POLE_MASS, HALF_LENGTH = 9.8, 1.0, 0.1, 0.5
FORCE, TAU, MAX_STEPS = 10.0, 0.02, 500
X_LIMIT, THETA_LIMIT = 2.4, 12 * 2 * math.pi / 360


class CartPole:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = FORCE if action == 1 else -FORCE
        cos, sin = math.cos(theta), math.sin(theta)
        total = CART_MASS + POLE_MASS
        temp = (force + POLE_MASS * HALF_LENGTH * theta_dot ** 2 * sin) / total
        theta_acc = (GRAVITY * sin - cos * temp) / (
            HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos ** 2 / total))
        x_acc = temp - POLE_MASS * HALF_LENGTH * theta_acc * cos / total
        self.state = np.array([x + TAU * x_dot, x_dot + TAU * x_acc,     # Euler step
                               theta + TAU * theta_dot, theta_dot + TAU * theta_acc])
        self.steps += 1
        x, theta = self.state[0], self.state[2]
        terminated = bool(abs(x) > X_LIMIT or abs(theta) > THETA_LIMIT)
        truncated = self.steps >= MAX_STEPS
        return self.state.copy(), 1.0, terminated, truncated


if __name__ == '__main__':
    env = CartPole(seed=0)
    rng = np.random.default_rng(0)
    lengths = []
    for _ in range(100):                               # a random policy, for scale
        env.reset()
        done, total = False, 0.0
        while not done:
            _, reward, terminated, truncated = env.step(int(rng.integers(2)))
            total += reward
            done = terminated or truncated
        lengths.append(total)
    print(f'random policy over 100 episodes: mean return {np.mean(lengths):.1f}, '
          f'max {max(lengths):.0f}')
