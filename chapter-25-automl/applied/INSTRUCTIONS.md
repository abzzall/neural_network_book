# Chapter 25 applied exercise — exhaustive search versus successive halving

## 1. Outcome

A search over ten MLP candidates for the digits (five hidden-layer shapes,
two weight penalties) done in two ways: every candidate trained for the full
200 epochs, and successive halving, which trains all candidates for 25 epochs,
keeps the better half, doubles the budget and repeats. Weights are fitted on
the training split and candidates ranked on the validation split; only the two
selected procedures are refitted on all development rows and scored once on
the untouched test split. Record what each search selects and what it costs.

## 2. Verified versions

Python 3.14.4, scikit-learn 1.9.1, 18 October 2026. No download is needed.
The full run takes a few minutes on a laptop CPU.

## 3. Steps

1. `candidates`: return every (shape, penalty) pair, shapes in the outer loop.
2. `exhaustive`: score every candidate after 200 epochs on the validation
   split; return the best (the first of any ties), its score and the cost,
   candidates times 200 epochs.
3. `successive_halving`: start with 25 epochs; after each round keep the best
   half (at least one), double the epochs (at most 200), and add the epochs
   spent to the cost; stop when one candidate is left or the budget is full.
4. `final_test`: refit the selected candidate on all development rows for 200
   epochs and return its test accuracy. Call it only for the selected
   procedures.
5. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-25-automl/applied/based-on
.venv/bin/python chapter-25-automl/applied/based-on/automl_search.py
```

## 4. Verification

```bash
.venv/bin/python chapter-25-automl/applied/full-code/verify_results.py chapter-25-automl/applied/RESULTS.md
```

Epoch counts must match exactly; record `same_selection` as 1 if both searches
select the same candidate, 0 otherwise. Accuracies may differ slightly across
platforms.

The `*_epochs` fields are allocated maximum candidate-epoch budgets, not measured
computation. Each round restarts fitting; convergence can use fewer epochs and
wide candidates cost more per epoch. The finished runner separately reports
`*_actual_epochs` and `*_fit_seconds` for search fits, excluding the final refit.
