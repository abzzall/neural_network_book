"""REINFORCE: a policy-gradient agent for the cart-pole task."""

import numpy as np
import torch
from torch import nn

from cartpole import CartPole


def returns_to_go(rewards, gamma):
    """G_t = r_{t+1} + gamma r_{t+2} + ..., eq. (19.1), for every step of an episode."""
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def train(seed, episodes=600, gamma=0.99, lr=1e-2):
    torch.manual_seed(seed)
    env = CartPole(seed=seed)
    policy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(episodes):
        state, done, log_probs, rewards = env.reset(), False, [], []
        while not done:                                  # play one episode
            dist = torch.distributions.Categorical(
                logits=policy(torch.tensor(state, dtype=torch.float32)))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated = env.step(int(action))
            rewards.append(reward)
            done = terminated or truncated
        g = torch.tensor(returns_to_go(rewards, gamma))
        advantage = (g - g.mean()) / (g.std() + 1e-8)   # mean as baseline, eq. (19.8)
        loss = -(torch.stack(log_probs) * advantage).sum()   # gradient of eq. (19.7)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        history.append(sum(rewards))
    return history, policy


def greedy(policy):
    return lambda s: int(policy(torch.tensor(s, dtype=torch.float32)).argmax())
