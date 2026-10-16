# Chapter 26 — Capstone Project

A reproducible project scaffold, five data-type tracks and a completed synthetic
classification example. The example compares linear and quadratic logistic
regression with two NumPy MLPs from the earlier chapters. It selects using
validation data and then reports a final test comparison.

Verified environment: Python 3.14.4, NumPy 2.5.4 and scikit-learn 1.9.1.
Use the existing repository environment; no new package or download is needed.
The original synthetic generator and examples use the repository MIT licence.

## Run the completed example

From the repository root:

```bash
.venv/bin/python chapter-26-capstone/project_demo.py
.venv/bin/python chapter-26-capstone/applied/full-code/capstone_project.py --output outputs/capstone/results.json
.venv/bin/python chapter-26-capstone/applied/full-code/verify_results.py outputs/capstone/results.json
```

The generated data have 1200 independent rows and six features. Split sizes
are 720/240/240. Scaling is fitted on training rows only. Candidate order is
linear, quadratic, MLP width 8, MLP width 16; the first candidate within 0.01
of the best mean validation macro-F1 is selected. Seeds are 42, 43 and 44;
neural training uses 160 epochs, batch size 64, learning rate 0.01 and
Adam with decoupled weight decay 0.001.

Measured results on the verified environment:

| Candidate | Validation macro-F1 mean | Population standard deviation |
| --- | ---: | ---: |
| Linear logistic | 0.691934 | 0.000000 |
| Quadratic logistic | 0.928390 | 0.000000 |
| MLP width 8 | 0.893230 | 0.008137 |
| MLP width 16 | 0.893310 | 0.008664 |

The selected quadratic model gives test macro-F1 0.895817; the linear baseline
gives 0.682783. The representative seed is always the first declared seed,
not the best one. Its selected-model error count is 25, including 16 errors
among 89 rows in the predeclared `abs(x0*x1) < 0.25` diagnostic slice.
Last digits can vary across numerical environments. Zero seed spread for the
deterministic logistic methods does not measure population uncertainty.

These are teaching-fixture results. They do not determine which model an
independent real-world project should use. Candidate configurations and data
are supplied for transparent reproduction, not as a blinded research benchmark.

## Evidence and acceptance

The JSON records data identity, environment, split IDs, audit counts, training
budget, per-seed validation/test metrics, selection rule and traceable errors.
The verifier checks consistency and compares a saved record with a fresh local
run. Metric tolerance is 0.01; identity, split, selected procedure and error
records must agree. The acceptance tests also perturb final test labels and
check that selection and validation stay unchanged.

The starter in [applied/based-on](applied/based-on) intentionally raises numbered
errors. See [exercise instructions](applied/INSTRUCTIONS.md) and
[results template](applied/RESULTS.md). Completed code is in
[applied/full-code](applied/full-code). Repository-wide checks exclude the starter.

For an independent capstone, use [PROJECT_BRIEF.md](PROJECT_BRIEF.md),
[TRACKS.md](TRACKS.md) and [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).
[rubric.json](rubric.json) supplies the proposed 100-point marking weights.
Adapt task-specific contracts to the chosen track; do not require other data
types to reproduce this binary fixture's exact counts or scores.
