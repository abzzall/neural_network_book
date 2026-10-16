# Chapter 7 applied exercise — comparing optimisers

## 1. Outcome

The digits network of chapter 6 trained for 30 epochs in four ways (full-batch
gradient descent, mini-batch SGD, SGD with momentum, Adam), compared by how
quickly the training loss falls below 0.05 and by test accuracy.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1 (2026-10-16).
Runs on a CPU in a few seconds.

## 3. Files supplied

- `based-on/compare.py` — the starting point; `load_split` and `build` are
  supplied, each other function raises `NotImplementedError` naming its step.
- `based-on/test_compare.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.

## 4. Steps

```bash
.venv/bin/python -m pytest chapter-07-optimization/applied/based-on
```

1. `settings`: full-batch GD (lr 0.5), mini-batch SGD (lr 0.05, batch 32),
   SGD with momentum 0.9 (lr 0.05, batch 32), Adam (lr 0.001, batch 32).
2. `train_all`: a fresh `build()` per setting, 30 epochs, seed 0.
3. `epochs_to`: the first epoch whose starting training loss is below 0.05.
4. `accuracy`: on the test samples.

## 5. Verification

```bash
.venv/bin/python chapter-07-optimization/applied/full-code/verify_results.py chapter-07-optimization/applied/RESULTS.md
```
