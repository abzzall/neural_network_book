# Chapter 19 applied exercise — a deep Q-network for the cart-pole task

## 1. Outcome

A deep Q-network (DQN) that learns to balance the pole from rewards alone. Each
of seeds 0, 1 and 2 trains for 100,000 environment steps; every 20,000 steps the
greedy policy is scored on 20 fresh episodes. Record where training ends and how
unstable the path there is, and compare with the REINFORCE results of
`reinforce_demo.py`.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 (CPU), pytest 9.1.1, 17 October 2026.
The complete solution takes about 4 minutes (one CPU thread per run). The
environment is `cartpole.py` in this chapter; nothing is downloaded.

## 3. Files supplied

- `based-on/dqn_cartpole.py`: incomplete; the network, training loop and
  evaluation are supplied.
- `based-on/test_dqn_cartpole.py`: complete acceptance tests (the last one
  trains all three seeds).
- `full-code/`: finished solution, tests and result verifier.
- `RESULTS.md`: values to record and questions to explain.

## 4. Steps

1. `epsilon`: exploration probability falling linearly from 1 to 0.05 over
   20,000 steps, then constant.
2. `Replay.add` and `Replay.sample`: store transitions in a ring buffer that
   overwrites the oldest entry when full; sample a minibatch uniformly from the
   stored entries.
3. `td_target`: `r + gamma * max_a Q_target(s', a)`, with no bootstrap term
   when the episode terminated. The 500-step time limit is not a termination:
   the pole was still up, so the future value must still be counted.
4. In the training loop, every 1,000 steps copy the online network's weights
   into the target network.
5. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-19-rl/applied/based-on -k "not full_run"
.venv/bin/python chapter-19-rl/applied/based-on/dqn_cartpole.py
```

## 5. What to record

The mean and minimum over seeds of the final greedy return, the number of
checkpoints (of 15) that scored the maximum 500, and the lowest checkpoint
score after the first one.

## 6. Verification

```bash
.venv/bin/python chapter-19-rl/applied/full-code/verify_results.py chapter-19-rl/applied/RESULTS.md
```

Every field must report `OK`. Reinforcement learning amplifies tiny numerical
differences, so the tolerances are wide.

## 7. Troubleshooting

- Returns stuck near 10: actions may never be greedy; check the epsilon schedule.
- Returns that rise and then collapse repeatedly: expected to some degree; if
  extreme, check that the target network is updated only every 1,000 steps.
- Returns that never exceed about 200: check that truncated episodes are not
  stored as terminated.
