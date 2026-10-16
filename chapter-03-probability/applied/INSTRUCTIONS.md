# Chapter 3 applied exercise — maximum likelihood two ways

## 1. Outcome

The click probability of 200 page views and the mean and variance of 200
delivery times, each estimated twice: by the maximum-likelihood formula and by
gradient descent on the negative log-likelihood.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, pytest 9.1.1 (2026-10-16). Runs on a CPU in under a second.

## 3. Files supplied

- `based-on/estimate.py` — the starting point; each function raises
  `NotImplementedError` naming its step.
- `based-on/test_estimate.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 4. Steps

Work in `based-on/estimate.py` and run the tests after each step:

```bash
.venv/bin/python -m pytest chapter-03-probability/applied/based-on
```

1. `bernoulli_mle`: the fraction of ones.
2. `bernoulli_nll`: the average binary cross-entropy, in nats.
3. `bernoulli_gd`: gradient descent on p from 0.5 (learning rate 0.05, 500 steps).
4. `gaussian_mle`: the mean, and the variance divided by n.
5. `gaussian_nll`: the average Gaussian negative log-likelihood, with s = log(variance).
6. `gaussian_gd`: gradient descent on (mu, s) from (0, 0) (learning rate 0.5, 2000 steps).

## 5. What to record

Run your functions on `make_data()` and fill in the table in `RESULTS.md`.
`var_unbiased` is `times.var(ddof=1)`.

## 6. What to explain

Answer the three questions in `RESULTS.md` in your own words.

## 7. Verification

```bash
.venv/bin/python chapter-03-probability/applied/full-code/verify_results.py chapter-03-probability/applied/RESULTS.md
```

Every line must start with `OK`.

## 8. Troubleshooting

- `RuntimeWarning: divide by zero in log` — p reached 0 or 1; lower the learning rate.
- The variance becomes negative — use s = log(variance), as step 5 says.
