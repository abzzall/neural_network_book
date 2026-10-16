"""Deep Q-network for the cart-pole task: replay buffer, target network, epsilon-greedy.

Each seed trains for 100,000 environment steps; the greedy policy is scored on
20 fresh episodes every 20,000 steps, so the evaluation shows the learning path.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cartpole import CartPole
from rl_utils import evaluate

STEPS, CHECK_EVERY, GAMMA, BATCH, SYNC = 100_000, 20_000, 0.99, 64, 1000


def q_network():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 2))


def epsilon(step, start=1.0, end=0.05, decay_steps=20_000):
    """Step 1: exploration falls linearly from 1 to 0.05, then stays there."""
    return max(end, start - (start - end) * step / decay_steps)


class Replay:
    """Step 2: a ring buffer of transitions and uniform random minibatches."""

    def __init__(self, size=50_000):
        self.s, self.s2 = np.zeros((size, 4), np.float32), np.zeros((size, 4), np.float32)
        self.a, self.r = np.zeros(size, np.int64), np.zeros(size, np.float32)
        self.end, self.n, self.size = np.zeros(size, np.float32), 0, size

    def add(self, s, a, r, s2, terminated):
        i = self.n % self.size
        self.s[i], self.a[i], self.r[i], self.s2[i], self.end[i] = s, a, r, s2, terminated
        self.n += 1

    def sample(self, rng, batch=BATCH):
        idx = rng.integers(min(self.n, self.size), size=batch)
        return [torch.tensor(v[idx]) for v in (self.s, self.a, self.r, self.s2, self.end)]


def td_target(target_net, r, s2, terminated):
    """Step 3: the TD target of eq. (19.6); no bootstrap after termination."""
    with torch.no_grad():
        return r + GAMMA * (1 - terminated) * target_net(s2).max(dim=1).values


def train(seed, steps=STEPS, warm_up=1000, lr=5e-4):
    torch.manual_seed(seed)
    rng, env = np.random.default_rng(seed), CartPole(seed=seed)
    q, target = q_network(), q_network()
    target.load_state_dict(q.state_dict())
    optimiser, replay = torch.optim.Adam(q.parameters(), lr=lr), Replay()
    greedy = lambda s: int(q(torch.tensor(s, dtype=torch.float32)).argmax())
    state, checks = env.reset(), []
    for step in range(1, steps + 1):
        if rng.random() < epsilon(step):
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = greedy(state)
        nxt, reward, terminated, truncated = env.step(action)
        replay.add(state, action, reward, nxt, terminated)   # not truncated
        state = env.reset() if terminated or truncated else nxt
        if replay.n >= warm_up:
            s, a, r, s2, end = replay.sample(rng)
            q_sa = q(s).gather(1, a[:, None])[:, 0]
            loss = nn.functional.smooth_l1_loss(q_sa, td_target(target, r, s2, end))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if step % SYNC == 0:                     # Step 4: copy the weights
            target.load_state_dict(q.state_dict())
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                checks.append(evaluate(greedy, seed))
    return checks


def run(seeds=(0, 1, 2)):
    torch.set_num_threads(1)
    checks = np.array([train(seed) for seed in seeds])        # (seeds, checkpoints)
    return {'final_mean': float(checks[:, -1].mean()), 'final_min': float(checks[:, -1].min()),
            'checkpoints_at_500': int((checks == 500).sum()),
            'worst_after_40k': float(checks[:, 1:].min()),
            'checks': checks.tolist()}


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
