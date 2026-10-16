# Chapter 16 applied exercise — a probabilistic forecast of bike rentals

## 1. Outcome

A linear model that forecasts the 10%, 50% and 90% quantiles of daily bike
rentals for each of the next 14 days, trained with the pinball loss. It is
compared with a simple interval built around the seasonal-naive forecast. Both
are evaluated at the same 25 weekly origins from 1 July 2012 by MASE of the
median, coverage of the 10%–90% interval, and mean pinball loss.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 (CPU), pytest 9.1.1, 16 October 2026.
The complete solution takes about 6 seconds.

## 3. Data (download once)

```bash
.venv/bin/python chapter-16-time-series/bike_data.py --download
```

Bike Sharing Dataset, H. Fanaee-T, UCI Machine Learning Repository,
doi 10.24432/C5W894, licensed CC BY 4.0. The file is checked against a SHA-256
hash and is not stored in this repository.

## 4. Files supplied

- `based-on/quantile_bikes.py`: incomplete; data loading, windows, the
  optimiser and its schedule, and `run()` are supplied.
- `based-on/test_quantile_bikes.py`: complete acceptance tests.
- `full-code/`: finished solution, tests and result verifier.
- `RESULTS.md`: values to record and questions to explain.

## 5. Steps

1. `pinball_loss`: for each level tau, the loss is `tau * (y - q)` when the
   actual value is above the quantile and `(tau - 1) * (y - q)` when it is
   below; return the mean over all elements.
2. `design`: stack the scaled past window, a column of ones, and the 14
   future working-day flags. Accept a single window as well as a batch.
3. In `train`: reshape the model output to (windows, 14, 3), compute the
   pinball loss, then `zero_grad`, `backward`, `step`.
4. `quantile_forecast`: build the inputs at origin t with `window`, predict,
   multiply by the window level, and sort the three values of each day so the
   quantiles never cross.
5. `naive_interval`: for each origin s in the last year of history, record the
   14 errors of the seasonal-naive forecast made at s; add the 10%, 50% and 90%
   quantiles of these errors to the current seasonal-naive forecast.
6. In `run`: score both methods at every origin with `scores`.
7. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-16-time-series/applied/based-on
.venv/bin/python chapter-16-time-series/applied/based-on/quantile_bikes.py
```

## 6. What to record

The number of training windows and origins, and for both methods the MASE of
the median, the coverage of the 10%–90% interval and the mean pinball loss.

## 7. Verification

```bash
.venv/bin/python chapter-16-time-series/applied/full-code/verify_results.py chapter-16-time-series/applied/RESULTS.md
```

Every field must report `OK`. Counts must match exactly; coverage may differ by
0.015 and other scores by 1% across platforms. Written explanations require
human assessment.

## 8. Troubleshooting

- `FileNotFoundError` naming `--download`: download the data first.
- `NotImplementedError`: complete the indicated step.
- Coverage near 0 or 1: check the sign convention in the pinball loss.
- Results that change a lot between runs: the training is seeded; check that
  `torch.manual_seed` runs before the model is built.
