# Chapter 5 applied exercise — logistic regression from scratch

## 1. Outcome

A logistic-regression classifier, trained by gradient descent from scratch,
that separates malignant from benign tumours in the Breast Cancer Wisconsin
(Diagnostic) data, compared with scikit-learn and interpreted through its
weights. This is a teaching exercise, not a diagnostic tool.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1 (2026-10-16).
Runs on a CPU in about a second.

## 3. Data

Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). Breast Cancer
Wisconsin (Diagnostic). UCI Machine Learning Repository,
https://doi.org/10.24432/C5DW2B. Licensed CC BY 4.0. Distributed with
scikit-learn as `load_breast_cancer`; no download is needed.

## 4. Files supplied

- `based-on/classify.py` — the starting point; each function after `load`
  raises `NotImplementedError` naming its step.
- `based-on/test_classify.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 5. Steps

Work in `based-on/classify.py` and run the tests after each step:

```bash
.venv/bin/python -m pytest chapter-05-linear-models/applied/based-on
```

1. `load` is supplied: malignant tumours are the positive class (label 1).
2. `prepare`: a stratified 75/25 split (seed 0) and standardisation fitted on the training rows.
3. `train`: `nnbook.LogisticRegression` with learning rate 0.1 for 2000 epochs.
4. `evaluate`: accuracy, missed malignant tumours, and false alarms on the test rows.
5. `reference_accuracy`: scikit-learn's `LogisticRegression(max_iter=5000)` on the same split.
6. `largest_weights`: the features with the largest absolute weights.

## 6. What to record

Fill in the table in `RESULTS.md`; `largest_weight_feature` is the name exactly
as printed (for example `mean radius`).

## 7. Verification

```bash
.venv/bin/python chapter-05-linear-models/applied/full-code/verify_results.py chapter-05-linear-models/applied/RESULTS.md
```

Every line must start with `OK`.

## 8. Troubleshooting

- The loss is `nan` — the learning rate is too large, or the features were not standardised.
- Accuracy near 0.63 — the model predicts one class for everything; check the label flip in `load`.
