# Chapter 12 applied exercise — what convolution buys

## 1. Outcome

A small convolutional network compared with the multilayer perceptron of
chapter 6 on the digits: parameters, test accuracy, and accuracy when every
test image is shifted one pixel to the right; then the same CNN trained with
shift augmentation.

## 2. Verified versions

Python 3.14.4, PyTorch 2.14.1, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1
(2026-10-16). Runs on a CPU in under a minute. Accuracies may differ in the last
digits on other machines.

## 3. Files supplied

- `../digits_data.py` — loads the images and provides `fit`, `accuracy`,
  `n_params`, and `shift_right`.
- `based-on/cnn_vs_mlp.py` — the starting point; each function you write raises
  `NotImplementedError` naming its step. `fit_augmented` is supplied.
- `based-on/test_cnn_vs_mlp.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.

## 4. Steps

```bash
.venv/bin/python -m pytest chapter-12-cnn/applied/based-on
```

1. `mlp`: `Flatten`, `Linear(64, 32)`, `ReLU`, `Linear(32, 10)`.
2. `cnn`: two stages of `Conv2d` (3 x 3, padding 1), `ReLU`, `MaxPool2d(2)` with
   16 and 32 channels, then `Flatten` and `Linear(128, 10)`.
3. `random_shift`: move a batch by -1, 0 or 1 pixel in each direction.
4. `evaluate`: the mean accuracy over seeds 0, 1, 2 on the test images and on
   the test images shifted one pixel to the right.

## 5. Verification

```bash
.venv/bin/python chapter-12-cnn/applied/full-code/verify_results.py chapter-12-cnn/applied/RESULTS.md
```
