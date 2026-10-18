# Chapter 8 — Regularisation and Model Evaluation

Executable material for chapter 8 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `curves.py` | — | Shared data (noisy sin 2πx), polynomial features, and a stable ridge solver |
| `bias_variance.py` | Figure data | Training and validation error of polynomials of degree 0 to 15 |
| `ridge_demo.py` | — | An L2 penalty tames a degree-12 polynomial |
| `cv_demo.py` | — | Five-fold cross-validation chooses the penalty |
| `pr_curve.py` | Worked example | Rare-event precision-recall and non-interpolated AP |
| `metrics_demo.py` | Worked example | Accuracy, precision, recall, F1 and ROC-AUC, checked against scikit-learn |
| `calibration_demo.py` | Figure data | An accurate but overconfident network, the Brier score and ECE, and Platt scaling |
| `compare_models.py` | — | Paired fold differences, a paired t-test and a bootstrap interval for two models |
| `conformal_demo.py` | — | Abstaining on uncertain digits, and conformal prediction sets |
| `test_scripts.py` | — | Runs every script and checks what it prints |
| `applied/` | Applied analysis | Overfitting on 200 digits, and four ways to regularise |

The metrics, cross-validation splits, dropout, and batch normalisation are in
[`nnbook/metrics.py`](../nnbook/metrics.py), [`nnbook/model_selection.py`](../nnbook/model_selection.py),
and [`nnbook/regularization.py`](../nnbook/regularization.py) (listing 8.1), tested in
[`tests/test_regularization.py`](../tests/test_regularization.py). Early stopping is the
`validation` and `patience` arguments of `Network.fit`.

```bash
.venv/bin/python chapter-08-regularization/bias_variance.py
.venv/bin/python -m pytest chapter-08-regularization
```

The unregularised high-degree fits are extremely ill-conditioned, so their last
digits can differ slightly between machines; the tests allow for this.

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.

## Numerical checks and interpretation

`SoftmaxCrossEntropy` uses log-sum-exp on logits, with an extreme-logit gradient
test in `tests/test_network.py`. Conformal thresholds use the exact finite-sample
order statistic and include every class when the requested rank exceeds the
calibration sample count. `compare_models.py` reports a paired bootstrap interval
against a prespecified one-percentage-point useful gain. Its overlapping-fold
t-test is a naive diagnostic, not a calibrated significance test.

```bash
.venv/bin/python chapter-08-regularization/pr_curve.py
.venv/bin/python chapter-08-regularization/compare_models.py
.venv/bin/python chapter-08-regularization/conformal_demo.py
```
