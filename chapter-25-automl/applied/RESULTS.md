# Chapter 25 applied exercise — results

| Quantity | Value |
|---|---|
| `candidates` | |
| `exhaustive_validation` | |
| `exhaustive_test` | |
| `exhaustive_epochs` | |
| `halving_validation` | |
| `halving_test` | |
| `halving_epochs` | |
| `same_selection` | |

## What to explain

1. Why can adding the candidate (128, 64, 32) never lower the best validation
   score, yet still lower the test score?
2. Successive halving allocated less than half the candidate-epochs. Why did it not miss the
   best candidate here, and when could it?
3. Which number would you report as the expected accuracy of the selected
   model, and why not the validation score?

The `*_epochs` fields are allocated maximum candidate-epoch budgets, not measured
computation. Each round restarts fitting; convergence can use fewer epochs and
wide candidates cost more per epoch. The finished runner separately reports
`*_actual_epochs` and `*_fit_seconds` for search fits, excluding the final refit.
