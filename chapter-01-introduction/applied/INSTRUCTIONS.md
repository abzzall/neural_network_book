# Chapter 1 applied exercise — what the bias of a neuron does

## 1. Outcome

The decisions of the irrigation neuron of chapter 1 on eight days, how they
change as the bias changes, and the exact bias at which each day's decision flips.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, pytest 9.1.1 (2026-10-16). Runs instantly.

## 3. Files supplied

- `based-on/neuron.py` — the starting point; the weights and the eight days are
  given, and each function raises `NotImplementedError` naming its step.
- `based-on/test_neuron.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 4. Steps

```bash
.venv/bin/python -m pytest chapter-01-introduction/applied/based-on
```

1. `weighted_sums`: the weighted sum of every day, without the bias.
2. `decisions`: 1 (irrigate) where the weighted sum plus the bias is positive.
3. `flip_bias`: the bias at which each day's weighted sum plus bias is exactly zero.
4. `irrigation_days`: how many days are irrigated for a given bias.

## 5. What to record

The number of irrigation days for biases 0.5, 1.0 and 1.5, and the flip bias of
the first and the last day.

## 6. Verification

```bash
.venv/bin/python chapter-01-introduction/applied/full-code/verify_results.py chapter-01-introduction/applied/RESULTS.md
```
