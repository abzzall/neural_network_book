"""Run the chapter's scripts as a reader would and check what they print."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, cwd):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=300)
    return result.stdout.splitlines()


def test_value_iteration(tmp_path):
    assert run('value_iteration.py', tmp_path) == [
        'after sweep 1:',
        ' 0.000  1.000    G  ',
        ' 0.000  0.000    P  ',
        ' 0.000  0.000  0.000',
        'after sweep 2:',
        ' 0.900  1.000    G  ',
        ' 0.000  0.900    P  ',
        ' 0.000  0.000  0.000',
        'converged after 5 sweeps:',
        ' 0.900  1.000    G  ',
        ' 0.810  0.900    P  ',
        ' 0.729  0.810  0.729',
        'greedy action from the start: up',
        'random policy, value of the start: -0.1700']


def test_q_learning(tmp_path):
    assert run('q_learning.py', tmp_path) == [
        'seed 0: error at start 0.0000, steps (last 50 episodes) 4.38, '
        'states off by > 0.01: [((2, 1), 25), ((2, 2), 8)]',
        'seed 1: error at start 0.0000, steps (last 50 episodes) 4.42, '
        'states off by > 0.01: [((1, 1), 71), ((2, 1), 77), ((2, 2), 35)]',
        'seed 2: error at start 0.0000, steps (last 50 episodes) 4.28, states off by > 0.01: []']


def test_cartpole_random_and_hand_controller(tmp_path):
    assert run('cartpole.py', tmp_path) == [
        'random policy over 100 episodes: mean return 21.6, max 72']
    assert run('hand_controller.py', tmp_path) == [
        f'seed {s}: mean return over 20 episodes = 500.0' for s in (0, 1, 2)]


def test_ppo_clip(tmp_path):
    assert run('ppo_clip.py', tmp_path) == [
        'advantage +1:',
        '  ratio 0.5: objective +0.50, gradient +1',
        '  ratio 0.9: objective +0.90, gradient +1',
        '  ratio 1.0: objective +1.00, gradient +1',
        '  ratio 1.1: objective +1.10, gradient +1',
        '  ratio 1.5: objective +1.20, gradient +0',
        'advantage -1:',
        '  ratio 0.5: objective -0.80, gradient +0',
        '  ratio 0.9: objective -0.90, gradient -1',
        '  ratio 1.0: objective -1.00, gradient -1',
        '  ratio 1.1: objective -1.10, gradient -1',
        '  ratio 1.5: objective -1.50, gradient -1']


def test_reinforce_demo(tmp_path):
    """Policy gradients learn the task for every seed, but not monotonically."""
    lines = run('reinforce_demo.py', tmp_path)
    assert len(lines) == 3
    for line in lines:
        assert 'solved at episode None' not in line
        assert line.split('greedy policy ')[1].startswith('500.0')
    lasts = [float(line.split('last 50 episodes ')[1].split(',')[0]) for line in lines]
    assert min(lasts) < 400 < max(lasts)          # one seed has fallen back after solving
