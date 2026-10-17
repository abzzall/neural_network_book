# Chapter 25 — Choosing a Model in Practice

Examples verified on a CPU with Python 3.14.4, scikit-learn 1.9.1 and pandas
3.0.6 (17 October 2026). Run commands from the repository root. The credit data
are downloaded by chapter 23; the other data sets are bundled with scikit-learn.

| File | What is checked |
|---|---|
| `family_compare.py` | Logistic regression, gradient boosting and an MLP on two tables and one image set, 5-fold cross-validation |
| `applied/` | A credit policy chosen by the data set's own cost matrix |

```bash
.venv/bin/python chapter-23-interpretability/credit_data.py --download
.venv/bin/python chapter-25-model-selection/family_compare.py
.venv/bin/python -m pytest chapter-25-model-selection --ignore-glob='*/based-on/*'
```

Expected output of `family_compare.py` (times depend on the machine):

```text
credit (table, 1000 rows), roc_auc:
  logistic regression    0.7913 +- 0.0238
  gradient boosting      0.7860 +- 0.0277
  neural network (MLP)   0.7550 +- 0.0334
breast cancer (table, 569 rows), roc_auc:
  logistic regression    0.9955 +- 0.0056
  gradient boosting      0.9940 +- 0.0053
  neural network (MLP)   0.9935 +- 0.0054
digits (8x8 images, 1797), accuracy:
  logistic regression    0.9694 +- 0.0099
  gradient boosting      0.9733 +- 0.0058
  neural network (MLP)   0.9716 +- 0.0113
```

## Applied exercise

```bash
.venv/bin/python chapter-25-model-selection/applied/full-code/credit_decision.py
.venv/bin/python chapter-25-model-selection/applied/full-code/verify_results.py
```

Reference output:

```text
bayes_threshold: 0.833333
approve_all_cost: 1.5
logistic_at_0.5_cost: 0.856
logistic_at_cost_threshold_cost: 0.48
logistic_at_cost_threshold_approved: 0.412
boosting_at_cost_threshold_cost: 0.641333
```
