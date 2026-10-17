# Chapter 25 applied exercise — choosing a credit policy by its costs

## 1. Outcome

The German Credit data come with a cost matrix: approving a bad risk costs 5,
rejecting a good one costs 1. Compare four policies by their mean cost per
applicant on test data, averaged over five random splits: approve everyone,
logistic regression with the default threshold 0.5, logistic regression with the
cost-optimal threshold, and gradient boosting with the same threshold.

## 2. Verified versions and data

Python 3.14.4, scikit-learn 1.9.1, pandas 3.0.6, 17 October 2026. A few seconds.
The data are those of chapter 23:

```bash
.venv/bin/python chapter-23-interpretability/credit_data.py --download
```

## 3. Steps

1. `mean_cost`: 5 for every approved bad risk, 1 for every rejected good risk,
   averaged over all applicants.
2. `bayes_threshold`: the probability of a good risk above which approving has the
   lower expected cost.
3. `policies`: fit both models on the training data and return the approval
   decisions of the four policies for the test applicants.
4. Run the tests and the script; fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-25-model-selection/applied/based-on
.venv/bin/python chapter-25-model-selection/applied/based-on/credit_decision.py
```

## 4. Verification

```bash
.venv/bin/python chapter-25-model-selection/applied/full-code/verify_results.py chapter-25-model-selection/applied/RESULTS.md
```

Values may differ by 0.02 across platforms. The exercise is a technical
illustration with an old public data set, not advice on credit decisions.
