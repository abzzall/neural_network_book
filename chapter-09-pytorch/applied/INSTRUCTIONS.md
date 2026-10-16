# Chapter 9 applied exercise — porting a network to PyTorch

## 1. Outcome

The dropout network of chapter 8 rewritten in PyTorch, trained with the same
settings, and compared with the nnbook version, taking the variation between
random seeds into account.

## 2. Verified versions

Python 3.14.4, PyTorch 2.14.1 (CPU), NumPy 2.5.4, scikit-learn 1.9.1, pytest
9.1.1 (2026-10-16). Runs on a CPU in under a minute. PyTorch does not promise
identical results on different machines; the last digits of your results may
differ from the book's.

## 3. Files supplied

- `based-on/port.py` — the starting point; `load_splits` is supplied, each other
  function raises `NotImplementedError` naming its step.
- `based-on/test_port.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.

## 4. Steps

```bash
.venv/bin/python -m pytest chapter-09-pytorch/applied/based-on
```

1. `build`: `Linear(64, 256)`, `ReLU`, `Dropout(0.5)`, `Linear(256, 10)`.
2. `train`: `torch.manual_seed(seed)`, a `DataLoader` with batches of 32 and a
   seeded generator, SGD with learning rate 0.05 and momentum 0.9, 200 epochs.
3. `accuracy`: `model.eval()` and `torch.inference_mode()`.
4. `seed_study`: the validation accuracy for seeds 0 to 4.

## 5. Verification

```bash
.venv/bin/python chapter-09-pytorch/applied/full-code/verify_results.py chapter-09-pytorch/applied/RESULTS.md
```
