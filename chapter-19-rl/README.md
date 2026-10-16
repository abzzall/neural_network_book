# Chapter 19 — Deep Reinforcement Learning

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4 and PyTorch 2.14.1
(17 October 2026). Run commands from the repository root. Every environment is
written in this folder; nothing is downloaded.

| File | What is checked |
|---|---|
| `gridworld.py`, `value_iteration.py` | A 3x3 grid world solved by policy evaluation and value iteration |
| `q_learning.py` | Tabular Q-learning from experience, compared with the Bellman solution |
| `cartpole.py` | The cart-pole task written out; matches Gymnasium CartPole-v1 step for step when Gymnasium is installed |
| `hand_controller.py` | A one-line controller from physical intuition |
| `reinforce.py`, `reinforce_demo.py` | REINFORCE with a mean baseline, three seeds |
| `ppo_clip.py` | PPO's clipped objective and its gradient |
| `rl_utils.py` | Policy evaluation and the "solved" criterion (mean of 100 returns at least 475) |
| `applied/` | A deep Q-network with replay buffer and target network, three seeds |

```bash
.venv/bin/python chapter-19-rl/value_iteration.py
.venv/bin/python chapter-19-rl/q_learning.py
.venv/bin/python chapter-19-rl/hand_controller.py
.venv/bin/python chapter-19-rl/reinforce_demo.py
.venv/bin/python -m pytest chapter-19-rl --ignore-glob='*/based-on/*'
```

Expected output of `value_iteration.py`:

```text
after sweep 1:
 0.000  1.000    G
 0.000  0.000    P
 0.000  0.000  0.000
after sweep 2:
 0.900  1.000    G
 0.000  0.900    P
 0.000  0.000  0.000
converged after 5 sweeps:
 0.900  1.000    G
 0.810  0.900    P
 0.729  0.810  0.729
greedy action from the start: up
random policy, value of the start: -0.1700
```

`reinforce_demo.py` solves the task (mean of 100 returns at least 475) at
episodes 190, 549 and 326 for seeds 0, 1 and 2, in about 45 seconds per seed;
its greedy policies all score 500. Learned results can differ across
platforms.

The comparison with Gymnasium is optional. With `gymnasium==1.4.0` (MIT
licence) installed, `test_rl.py` replays random and controlled episodes in both
implementations and requires identical states, rewards and end flags.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`.

```bash
.venv/bin/python chapter-19-rl/applied/full-code/dqn_cartpole.py
.venv/bin/python chapter-19-rl/applied/full-code/verify_results.py
```

Reference output:

```text
final_mean: 500
final_min: 500
checkpoints_at_500: 3
worst_after_40k: 14.3
```
