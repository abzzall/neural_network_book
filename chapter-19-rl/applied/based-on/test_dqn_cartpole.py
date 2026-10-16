"""Check the exploration schedule, replay buffer, TD target, and a full training run."""

import numpy as np
import pytest
import torch


def test_epsilon_schedule(load_script):
    m = load_script('dqn_cartpole')
    assert m.epsilon(0) == 1.0 and m.epsilon(10_000) == pytest.approx(0.525)
    assert m.epsilon(20_000) == pytest.approx(0.05) and m.epsilon(90_000) == 0.05


def test_replay_wraps_and_samples(load_script):
    m = load_script('dqn_cartpole')
    replay = m.Replay(size=3)
    for i in range(5):
        replay.add(np.full(4, i), i % 2, 1.0, np.full(4, i + 1), i == 4)
    assert replay.n == 5 and sorted(replay.s[:, 0]) == [2, 3, 4]   # oldest overwritten
    s, a, r, s2, end = replay.sample(np.random.default_rng(0), batch=8)
    assert s.shape == (8, 4) and set(s[:, 0].tolist()) <= {2.0, 3.0, 4.0}
    assert torch.equal(s2[:, 0], s[:, 0] + 1)


def test_td_target_stops_only_at_termination(load_script):
    m = load_script('dqn_cartpole')
    net = torch.nn.Linear(4, 2)
    torch.nn.init.zeros_(net.weight)
    torch.nn.init.constant_(net.bias, 10.0)                  # Q_target = 10 everywhere
    r, s2 = torch.tensor([1.0, 1.0]), torch.zeros(2, 4)
    target = m.td_target(net, r, s2, torch.tensor([0.0, 1.0]))
    assert target.tolist() == pytest.approx([1 + 0.99 * 10, 1.0])


def test_full_run_reaches_the_reference(load_script):
    m = load_script('dqn_cartpole')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
