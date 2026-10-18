# Chapter 4 — Data: Requirements, Cleaning, and Preprocessing

Executable material for chapter 4 of the book. Run every command from the
repository root. The listings table is **simulated** (`messy_data.py`), so every
defect in it is known exactly; no external dataset is downloaded.

| File | Book | What it shows |
|---|---|---|
| `messy_data.py` | — | Generates 515 apartment listings with inconsistent text, missing values, units mistakes, duplicates, and a leaking column |
| `cleaning_demo.py` | Listing 4.1 | Cleaning a small table step by step with pandas |
| `scaling_gd.py` | Figure data | Gradient-descent steps with a raw, centred, and standardised feature |
| `feature_selection_demo.py` | — | Correlation misses a nonlinear feature; mutual information finds it |
| `pca_demo.py` | — | Principal components from the covariance matrix, checked against scikit-learn |
| `leakage_demo.py` | Listing 4.2 | Feature selection on all the data makes pure noise look predictive |
| `imbalance_demo.py` | — | Plain versus stratified splits of a 1-in-100 class, and class weights |
| `pipeline_demo.py` | Listing 4.3 | The whole preparation as one scikit-learn pipeline |
| `kmeans_demo.py` | Clustering | k-means from scratch on simulated customers; scaling and choosing k |
| `annotation_agreement.py` | Worked example | Raw agreement and Cohen's kappa for two annotators |
| `test_scripts.py` | — | Runs every script and checks every printed line |
| `applied/` | Applied analysis | Cleaning, preparing, and fitting the listings; measuring a leak |

The from-scratch versions of the scalers, imputer, encoder, and split are in
[`nnbook/preprocessing.py`](../nnbook/preprocessing.py), tested against
scikit-learn in [`tests/test_preprocessing.py`](../tests/test_preprocessing.py).

```bash
.venv/bin/python chapter-04-data/leakage_demo.py
.venv/bin/python -m pytest chapter-04-data
```

Expected output of `leakage_demo.py`:

```text
mean validation R^2 over 50 datasets of pure noise:
  features chosen on all data:      0.419
  features chosen on training data: -0.314
```

`scaling_gd.py OUTPUT_DIR` writes the loss curves drawn in the book's figure.

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
