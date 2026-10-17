# Chapter 23 applied exercise — a fairness audit of a credit model

## 1. Outcome

A logistic-regression credit model predicts whether an applicant is a good
credit risk; a predicted good risk counts as an approval. Audit it for
applicants aged 25 or less: approval rates, true-positive rates (good risks
approved) and false-positive rates (bad risks approved) per group, and the
gaps between groups. Then compare two responses: removing age from the inputs,
and a separate approval threshold for young applicants that equalises
true-positive rates. Every number is a mean over five random splits.

## 2. Verified versions and data

Python 3.14.4, scikit-learn 1.9.1, pandas 3.0.6, 17 October 2026. About 2 seconds.

```bash
.venv/bin/python chapter-23-interpretability/credit_data.py --download
```

Statlog (German Credit Data), H. Hofmann, UCI Machine Learning Repository,
doi 10.24432/C5NC77, CC BY 4.0. Not stored in this repository.

## 3. Steps

1. `rates`: for each group (0 = older, 1 = young), the share approved, the
   share of good risks approved (TPR) and the share of bad risks approved (FPR).
2. `gaps`: young minus older for approval (parity gap) and TPR (opportunity
   gap); the odds gap is the larger absolute gap of TPR and FPR.
3. `young_threshold`: the score threshold at which young good applicants in the
   validation data are approved at the target TPR (a quantile of their scores).
4. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-23-interpretability/applied/based-on
.venv/bin/python chapter-23-interpretability/applied/based-on/fairness_audit.py
```

## 4. Verification

```bash
.venv/bin/python chapter-23-interpretability/applied/full-code/verify_results.py chapter-23-interpretability/applied/RESULTS.md
```

Values may differ by 0.01 across platforms. This exercise is a technical audit
of an old public data set; it is not legal advice about any credit decision.
