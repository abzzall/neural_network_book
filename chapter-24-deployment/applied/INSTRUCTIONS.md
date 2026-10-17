# Chapter 24 applied exercise — monitoring a deployed model for drift

## 1. Outcome

A monitor for the digit classifier of this chapter. It compares two quantities
in production with their values on the training images: each image's mean pixel
intensity (an input monitor) and the model's confidence, its largest class
probability (an output monitor). Five production scenarios are simulated:
unchanged images, images shifted one pixel, fainter scans, only digits 0-4, and
a change of meaning in which the labels 1 and 7 are swapped. Record when each
monitor raises an alarm, and compare with the model's real accuracy.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 (CPU), SciPy 1.18.1 (for a test),
17 October 2026. No download and no optional package is needed; the model is
trained in a few seconds on first use and cached in `outputs/`.

## 3. Steps

1. `psi`: bins at the deciles of the reference; with reference shares r and
   current shares c per bin, return the sum of (c - r) ln(c / r). Clip shares
   below 1e-6 to avoid the logarithm of zero.
2. `ks_statistic`: the largest absolute difference between the two empirical
   distribution functions, evaluated at every observed value.
3. `alarm`: true if PSI exceeds 0.2 or the KS statistic exceeds
   1.628 * sqrt((n + m) / (n m)), its critical value at the 1% level.
4. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-24-deployment/applied/based-on
.venv/bin/python chapter-24-deployment/applied/based-on/drift_monitor.py
```

## 4. Verification

```bash
.venv/bin/python chapter-24-deployment/applied/full-code/verify_results.py chapter-24-deployment/applied/RESULTS.md
```

Alarms must match exactly; other values may differ slightly across platforms.
Record alarms as 1 (raised) or 0 (not raised).
