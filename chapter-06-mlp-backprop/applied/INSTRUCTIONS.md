# Chapter 6 applied exercise — a multilayer network from scratch

## 1. Outcome

A network with one hidden layer of 32 ReLU units, built from the `nnbook`
layers, checked against central differences, and trained by gradient descent
to recognise handwritten digits better than the softmax regression of chapter 5.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1 (2026-10-16).
Runs on a CPU in a few seconds.

## 3. Data

Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of Handwritten Digits.
UCI Machine Learning Repository, https://doi.org/10.24432/C50P49. Licensed
CC BY 4.0. Distributed with scikit-learn as `load_digits`.

## 4. Files supplied

- `based-on/mlp_digits.py` — the starting point; `load_split` is supplied and each
  other function raises `NotImplementedError` naming its step.
- `based-on/test_mlp_digits.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 5. Steps

```bash
.venv/bin/python -m pytest chapter-06-mlp-backprop/applied/based-on
```

1. `build`: 64 inputs, 32 ReLU units, 10 scores, softmax cross-entropy (seed 0).
2. `gradient_check`: `max_relative_error` on the first five training samples, before training.
3. `train`: full-batch gradient descent, learning rate 0.5, 300 epochs.
4. `accuracy`: on the training and on the test samples.
5. `mistakes`: every wrong test prediction as (index, true, predicted).

## 6. What to record

Fill in the table in `RESULTS.md`. For the gradient check write `yes` if the
error is below 1e-5.

## 7. Verification

```bash
.venv/bin/python chapter-06-mlp-backprop/applied/full-code/verify_results.py chapter-06-mlp-backprop/applied/RESULTS.md
```

## 8. Troubleshooting

- The gradient check fails — check the order of the factors in `dW` and the transpose in the
  gradient passed to the previous layer.
- The loss stops falling near 2.3 — the learning rate is too small, or the features are not standardised.
