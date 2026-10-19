"""Unit checks: grid world, Bellman solutions, Q-learning, cart-pole, returns, PPO."""

from pathlib import Path
import importlib.util
import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules
GYM = importlib.util.find_spec('gymnasium') is not None


def test_grid_moves_walls_and_rewards(load_script):
    g = load_script('gridworld')
    assert g.step((0, 1), 'right') == ((0, 2), 1.0, True)
    assert g.step((0, 2 - 1), 'down') == ((1, 1), 0.0, False)
    assert g.step((2, 2), 'up') == ((1, 2), -1.0, True)
    assert g.step((0, 0), 'left') == ((0, 0), 0.0, False)        # wall


def test_value_iteration_matches_shortest_paths(load_script):
    vi = load_script('value_iteration')
    V, sweeps = vi.value_iteration()
    # optimal value = gamma ** (steps to the goal - 1), avoiding the pit
    for state, steps in {(0, 1): 1, (0, 0): 2, (1, 1): 2, (1, 0): 3, (2, 1): 3,
                         (2, 0): 4, (2, 2): 4}.items():
        assert V[state] == pytest.approx(0.9 ** (steps - 1))
    assert sweeps == 5
    assert vi.greedy_policy(V)[(2, 2)] == 'left'                # never into the pit


def test_random_policy_satisfies_bellman_expectation(load_script):
    vi = load_script('value_iteration')
    g = load_script('gridworld')
    V = vi.evaluate_random_policy()
    for s in g.STATES:
        if s not in g.TERMINAL:
            assert V[s] == pytest.approx(np.mean([vi.backup(V, s, a) for a in g.ACTIONS]))


def test_q_learning_finds_the_optimal_start_value(load_script):
    q = load_script('q_learning')
    vi = load_script('value_iteration')
    V, _ = vi.value_iteration()
    for seed in (0, 1, 2):
        Q, lengths, visits = q.q_learning(seed=seed)
        assert Q[(2, 0)].max() == pytest.approx(V[(2, 0)], abs=1e-3)
        assert np.mean(lengths[-50:]) < 5


def test_returns_to_go(load_script):
    r = load_script('reinforce')
    assert r.returns_to_go([1, 1, 1], 0.5) == [1.75, 1.5, 1.0]


@pytest.mark.parametrize('theta', [-2.0, 0.0, 1.5])
@pytest.mark.parametrize('gamma', [0.5, 1.0])
def test_discounted_policy_gradient_matches_exact_objective(load_script, theta, gamma):
    m = load_script('policy_gradient_check')
    h = 1e-5
    numerical = (m.objective(theta + h, gamma) - m.objective(theta - h, gamma)) / (2 * h)
    assert m.expected_estimate(theta, gamma) == pytest.approx(numerical, abs=1e-10)
    assert m.expected_estimate(theta, gamma, baseline=(0.7, -0.3)) == pytest.approx(numerical)
    unweighted = m.expected_estimate(theta, gamma, outer_discount=False)
    assert unweighted == pytest.approx(numerical / gamma)


def test_solved_at(load_script):
    u = load_script('rl_utils')
    # the mean reaches 475 once 95 of the last 100 returns are 500: (5*10 + 95*500)/100
    assert u.solved_at([10] * 50 + [500] * 100) == 145
    assert u.solved_at([10] * 200) is None


def test_ppo_clip_by_hand(load_script):
    p = load_script('ppo_clip')
    r = torch.tensor([0.5, 1.0, 1.5])
    assert p.clipped_objective(r, torch.tensor(1.0)).tolist() == pytest.approx([0.5, 1.0, 1.2])
    assert p.clipped_objective(r, torch.tensor(-1.0)).tolist() == pytest.approx([-0.8, -1.0, -1.5])


def test_cartpole_upright_and_limits(load_script):
    c = load_script('cartpole')
    env = c.CartPole(seed=0)
    env.reset()
    env.state = np.zeros(4)
    state, reward, terminated, truncated = env.step(1)
    # pushing right from rest: cart accelerates right, pole falls left
    assert state[1] > 0 and state[3] < 0 and reward == 1.0 and not terminated
    env.state = np.array([0.0, 0.0, 0.21, 0.0])
    assert env.step(1)[2]                                        # past 12 degrees
    assert c.THETA_LIMIT == pytest.approx(12 * math.pi / 180)


@pytest.mark.skipif(not GYM, reason='optional: gymnasium is not installed')
def test_cartpole_matches_gymnasium(load_script):
    import gymnasium as gym
    c = load_script('cartpole')
    env, mine = gym.make('CartPole-v1'), c.CartPole()
    rng = np.random.default_rng(1)
    for episode in range(10):
        obs, _ = env.reset(seed=episode)
        mine.reset()
        mine.state = env.unwrapped.state.astype(float).copy()
        done = False
        while not done:                     # a simple controller, so some reach 500
            a = int(obs[2] + 0.5 * obs[3] > 0) if episode % 2 else int(rng.integers(2))
            obs, r1, te1, tr1, _ = env.step(a)
            s, r2, te2, tr2 = mine.step(a)
            np.testing.assert_allclose(obs, s.astype(np.float32), atol=1e-6)
            assert (r1, te1, tr1) == (r2, te2, tr2)
            done = te1 or tr1
