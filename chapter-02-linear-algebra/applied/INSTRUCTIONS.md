# Chapter 2 applied exercise — linear regression two ways

## 1. Outcome

A linear-regression model fitted to 50 noisy points twice: by gradient descent
and by the normal equations, with the gradient checked numerically and the
learning-rate bound computed from the eigenvalues of the Hessian.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, pytest 9.1.1 (2026-10-16). Runs on a CPU in under a second.

## 3. Files supplied

- `based-on/linreg.py` — the starting point. Each function raises
  `NotImplementedError` naming the step that completes it.
- `based-on/test_linreg.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 4. Steps

Work in `based-on/linreg.py`. After each step, run the tests from the
repository root:

```bash
.venv/bin/python -m pytest chapter-02-linear-algebra/applied/based-on
```

1. `design_matrix`: stack a column of ones and `x`.
2. `mse`: the mean squared error.
3. `gradient`: the gradient of the mean squared error.
4. Run the tests: `test_gradient_matches_central_differences` checks step 3.
5. `learning_rate_bound`: 2 divided by the largest eigenvalue of the Hessian.
   Also record the smallest and largest eigenvalues (`np.linalg.eigvalsh`).
6. `gradient_descent`: repeat the update from `theta = 0`.
7. `normal_equations`: solve the normal equations with `np.linalg.solve`.

## 5. What to record

Run your code with the data from `make_data()` and a learning rate of 0.9
times the bound, and fill in the table in `RESULTS.md`.

## 6. What to explain

Answer the three questions in `RESULTS.md` in your own words.

## 7. Verification

```bash
.venv/bin/python chapter-02-linear-algebra/applied/full-code/verify_results.py chapter-02-linear-algebra/applied/RESULTS.md
```

Every line must start with `OK`.

## 8. Troubleshooting

- `ValueError: matmul: Input operand ... has a mismatch` — a vector has the
  wrong shape; print `X.shape` and `theta.shape`.
- The loss becomes `inf` or `nan` — the learning rate is above the bound.
