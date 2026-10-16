# Chapter 13 applied exercise — forecast against a seasonal baseline

## 1. Outcome

A GRU that predicts the next value of a synthetic seasonal demand series by
learning a correction to the value observed seven days earlier. Compare it
with that seasonal baseline on chronological validation and test intervals.
The complete series is generated locally; there is no external dataset.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 and pytest 9.1.1, CPU,
16 October 2026. Use the environment described in chapter 9.

## 3. Files supplied

- `based-on/recurrent_forecast.py`: incomplete preprocessing and forward pass;
  the data generator, training loop and checkpoint selection are supplied.
- `based-on/test_recurrent_forecast.py`: complete acceptance tests.
- `full-code/`: finished solution, tests and result verifier.
- `RESULTS.md`: values to record and questions to explain.

## 4. Steps

Complete the numbered functions in `based-on/recurrent_forecast.py`:

1. `training_statistics`: compute the mean and standard deviation of only
   `series[:260]`; later observations must not affect the scaler.
2. `history_windows`: for each origin t, stack the normalised observations
   `series[t-lookback:t]`. The target at t must be excluded.
3. `seasonal_baseline`: return `series[origins-7]`.
4. In `Forecaster.forward`, run `self.recurrent(x)` and unpack its sequence
   output and final state; x has shape (batch, time, feature).
5. Map the last time-step output through `self.output` and squeeze its final
   dimension so that the correction has one value per window.
6. Return the baseline plus `self.scale` times the correction, putting the
   prediction back into the original units.
7. Run the acceptance tests, then run the script to record the comparison.
   Targets before 260 train the model; 260..339 select a checkpoint; 340..419
   are the held-out test. The training loop is supplied rather than rewritten.
8. Fill the results table and answer its questions.

```bash
.venv/bin/python -m pytest chapter-13-rnn/applied/based-on
.venv/bin/python chapter-13-rnn/applied/based-on/recurrent_forecast.py
```

## 5. What to record

Record the number of training windows, selected epoch for seed 42, baseline
errors, and the mean and population standard deviation of GRU mean absolute
errors over seeds 42, 43 and 44 on both later intervals. Keep four significant figures.
The experiment is rolling one-step forecasting: an earlier test observation
can enter a later test window after it has been observed. It is not an
80-step forecast made before any test observations are known.

## 6. What to explain

Answer the questions in `RESULTS.md`. Three seeds on a single synthetic series are an implementation demonstration,
not a model-selection recommendation.

## 7. Verification

```bash
.venv/bin/python chapter-13-rnn/applied/full-code/verify_results.py chapter-13-rnn/applied/RESULTS.md
```

Every field must report `OK`; the process then exits with status zero. Without
a path argument, the verifier prints reference values. It allows a small
numerical tolerance for differences across platforms. Written explanations
require human assessment.

## 8. Troubleshooting

- `NotImplementedError`: complete the indicated step.
- Error about GRU input dimensions: retain the feature dimension of length one.
- Bad errors despite decreasing training loss: check scaler fitting, slice end
  points, baseline units and whether the best validation checkpoint was restored.
- Better test results after repeated tuning: the test is now influencing model
  selection; start a new held-out evaluation before claiming an improvement.
