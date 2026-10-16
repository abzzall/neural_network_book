# Chapter 8 applied exercise — overfitting and regularisation

## 1. Outcome

A network with 256 hidden units trained on only 200 digit images, which
overfits, and four regularised versions of it (L2, dropout, early stopping,
batch normalisation), compared on a validation set; the best is then tested once.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1 (2026-10-16).
Runs on a CPU in a few seconds.

## 3. Files supplied

- `based-on/regularise.py` — the starting point; `load_splits` and `accuracy` are
  supplied, each other function raises `NotImplementedError` naming its step.
- `based-on/test_regularise.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.

## 4. Steps

```bash
.venv/bin/python -m pytest chapter-08-regularization/applied/based-on
```

1. `build`: `Dense(64, 256)`, optional `BatchNorm(256)`, `ReLU`, optional `Dropout(p)`, `Dense(256, 10)`.
2. `settings`: SGD with learning rate 0.05 and momentum 0.9 for all; baseline;
   L2 with `weight_decay=0.01`; dropout 0.5; early stopping with patience 10 on
   the validation set; batch normalisation.
3. `train_all`: up to 200 epochs, batches of 32, seed 0.
4. `choose`: the setting with the highest validation accuracy.

## 5. Verification

```bash
.venv/bin/python chapter-08-regularization/applied/full-code/verify_results.py chapter-08-regularization/applied/RESULTS.md
```
