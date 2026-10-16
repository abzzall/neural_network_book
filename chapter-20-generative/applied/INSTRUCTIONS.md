# Chapter 20 applied exercise — anomaly detection with an autoencoder

## 1. Outcome

An alarm that flags unusual images. The digit 3 plays the role of a defect:
models learn only from the other digits, and an image is flagged when its
reconstruction error exceeds a threshold. Compare an autoencoder with PCA (a
linear autoencoder) by recall, false-alarm rate and precision, and estimate the
precision that the same alarm would have if defects made up only 1% of items.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1, PyTorch 2.14.1 (CPU),
pytest 9.1.1, 17 October 2026. The complete solution takes about 10 seconds.
The digits dataset is bundled with scikit-learn (UCI, CC BY 4.0); nothing is
downloaded.

## 3. Files supplied

- `based-on/anomaly_threshold.py`: incomplete; the models and `run()` are supplied.
- `based-on/test_anomaly_threshold.py`: complete acceptance tests.
- `full-code/`: finished solution, tests and result verifier.
- `RESULTS.md`: values to record and questions to explain.

## 4. Steps

1. `split`: scale pixels to 0..1; split the normal digits 60/20/20 into
   training, validation and test sets with the given seed; put every digit 3
   into the test set.
2. `threshold`: return the 95th percentile of the validation scores.
3. `rates`: an alarm is a score above the threshold; return recall (share of
   anomalies with an alarm), false-alarm rate (share of normal items with an
   alarm) and precision (share of alarms that are anomalies).
4. `precision_at`: the precision if anomalies made up a share `prevalence` of
   all items, from recall and false-alarm rate.
5. In `run`: for both models, compute the threshold from validation scores and
   the rates on the test set.
6. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-20-generative/applied/based-on
.venv/bin/python chapter-20-generative/applied/based-on/anomaly_threshold.py
```

## 5. Verification

```bash
.venv/bin/python chapter-20-generative/applied/full-code/verify_results.py chapter-20-generative/applied/RESULTS.md
```

Every field must report `OK`. Counts must match exactly; rates may differ by
10% (relative) or 0.01 across platforms. Written explanations require human
assessment.

## 6. Troubleshooting

- `NotImplementedError`: complete the indicated step.
- False-alarm rate far from 5%: the threshold must come from validation
  normals, never from the test set.
- Precision above recall at 1% prevalence: check the formula; with rare
  anomalies most alarms are false.
