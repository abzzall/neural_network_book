# Complete and audit the capstone reference project

## Outcome

Complete five functions in a reproducible binary classification project, verify
its saved evidence and write a recommendation. Then specify an independent
capstone using one of the five tracks.

## Verified versions

Python 3.14.4, NumPy 2.5.4 and scikit-learn 1.9.1 on CPU, 2026-10-16.
Use the existing repository environment; no download or new dependency is needed.
The original synthetic data generator and examples use the repository MIT licence.

## Files supplied

`based-on/capstone_project.py` supplies data generation, fitting, the experiment
sequence and JSON output. Its first five functions are missing. The tests are
complete. `full-code/` contains the solution and results verifier. The chapter's
root supplies the shared model helper, evidence validator, project brief, track
descriptions, report template and marking weights.

## Steps

1. In `partition_ids`, permute row indices with `np.random.default_rng(seed)`;
   return the first three fifths, next fifth and remainder as train/validation/test.
2. In `fit_scaler`, return `StandardScaler().fit(x[train_ids])`.
3. In `macro_f1`, compute F1 separately for classes 0 and 1 and return their
   mean as a float. Each F1 is `2*TP/(2*TP+FP+FN)`, or zero for a zero denominator.
4. In `choose_candidate`, return the first candidate in `CANDIDATES` within
   `tolerance` of the best mean validation score. Use no test scores.
5. In `error_rows`, return dictionaries with `row_id`, `actual`, `predicted`
   only for mistakes, preserving input order.
6. Run the tests, project and verifier; save the JSON record.
7. Fill in the results and report templates and explain the selection.
8. Choose a track and complete the project brief before independent final testing.

The missing functions are numbered; each error names the action to implement.
The runner supplies three seeds, four candidates, the fixed neural training
budget and the first-seed error analysis. The quadratic baseline learns
second-degree features; a neural win is not an acceptance requirement.

## What to record

Save the complete JSON in `outputs/capstone/results.json` relative to the
repository root. Record split counts, validation means/spreads, selection,
final baseline/selected scores, representative error count and slice counts.
Keep the data fingerprint, split IDs, versions, seeds and budget in that JSON.
Record the source revision and working-tree state in the report.

## What to explain

- Why does the selected candidate follow the declared validation rule?
- Why is quadratic logistic regression a useful comparator for this generator?
- What does zero baseline seed spread measure, and what does it omit?
- What do the error slice and the synthetic data fail to establish?
- Which split and metric contracts must change for your independent track?

## Verification

From the repository root:

```bash
.venv/bin/python -m pytest chapter-26-capstone/applied/based-on/test_capstone_project.py
.venv/bin/python chapter-26-capstone/applied/based-on/capstone_project.py --output outputs/capstone/results.json
.venv/bin/python chapter-26-capstone/applied/full-code/verify_results.py outputs/capstone/results.json
```

The untouched starter fails intentionally. Filling its functions gives the
completed implementation. The verifier checks internal consistency and data,
split, selection, scaling and error identity against a fresh local run. Per-seed
scores may differ by at most 0.01; record unrounded scores to keep means/spreads
consistent. The reference-specific numerical tests do not define acceptable
accuracy for an independent project.

## Troubleshooting

Numbered errors identify incomplete functions. Fit the scaler on indexed
training rows, not on the whole array. Return stable row IDs for mistakes.
Check both classes explicitly in macro-F1. Keep candidate order and tie policy
fixed. JSON output belongs on stdout; an optional output argument saves the same
record. Do not change data or budget and then compare with the fixture's stored
expected results as if they were the same experiment.
