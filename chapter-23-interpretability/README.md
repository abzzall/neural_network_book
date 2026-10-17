# Chapter 23 — Interpretability and Fairness

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1,
SciPy 1.18.1 and PyTorch 2.14.1 (17 October 2026). Run commands from the
repository root.

| File | What is checked |
|---|---|
| `shapley_exact.py` | Exact Shapley values by enumerating coalitions; worked example and axioms |
| `credit_data.py`, `credit_model.py` | German Credit data (hash-checked download) and a logistic-regression model |
| `permutation_importance.py` | Permutation importance from scratch against scikit-learn |
| `proxy_check.py` | How well the other features reveal whether an applicant is young |
| `integrated_gradients.py` | Saliency and integrated gradients on a digit classifier: completeness and a sanity check |
| `applied/` | A fairness audit with two mitigations |

```bash
.venv/bin/python chapter-23-interpretability/shapley_exact.py
.venv/bin/python chapter-23-interpretability/integrated_gradients.py
.venv/bin/python chapter-23-interpretability/credit_data.py --download
.venv/bin/python chapter-23-interpretability/permutation_importance.py
.venv/bin/python -m pytest chapter-23-interpretability --ignore-glob='*/based-on/*'
```

Expected output of `shapley_exact.py`:

```text
Shapley values: 2.0000 3.0000 3.0000
sum 8.0000 = f(x) - f(baseline) = 8.0000
v({}) = 0.0
v({0}) = 2.0
v({1}) = 0.0
v({2}) = 0.0
v({1, 2}) = 6.0
v({0, 1, 2}) = 8.0
```

## Data

**Statlog (German Credit Data)** (1000 applicants, 20 attributes), H. Hofmann,
UCI Machine Learning Repository, doi 10.24432/C5NC77, licensed CC BY 4.0.
Downloaded by `credit_data.py` and checked against a SHA-256 hash; not stored
here. The digits are bundled with scikit-learn (UCI, CC BY 4.0).

## Applied exercise

```bash
.venv/bin/python chapter-23-interpretability/applied/full-code/fairness_audit.py
.venv/bin/python chapter-23-interpretability/applied/full-code/verify_results.py
```

Reference output:

```text
with_age_accuracy: 0.750667
with_age_older_tpr: 0.889456
with_age_young_tpr: 0.775203
with_age_opportunity_gap: -0.114253
with_age_odds_gap: 0.143894
without_age_opportunity_gap: -0.09413
group_threshold_accuracy: 0.75
group_threshold_opportunity_gap: -0.00221325
group_threshold_young_fpr: 0.716504
group_threshold_odds_gap: 0.209032
```
