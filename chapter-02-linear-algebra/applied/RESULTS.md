# Chapter 2 applied exercise — results

Fill in the second column with the values your code prints. Round to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-02-linear-algebra/applied/full-code/verify_results.py chapter-02-linear-algebra/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `lambda_min` | |
| `lambda_max` | |
| `learning_rate_bound` | |
| `b_after_100` | |
| `w_after_100` | |
| `b_gd` | |
| `w_gd` | |
| `b_normal` | |
| `w_normal` | |
| `mse_min` | |
| `mse_above_bound_exceeds_1e6` | |

## What to explain

1. After 100 steps the slope `w` is already close to its final value but the
   intercept `b` is not. Using the eigenvalues of the Hessian, explain why.
2. Why does the loss grow without limit when the learning rate is 10 per cent
   above the bound, although each step still moves against the gradient?
3. For a model with a million parameters, which of the two methods would you
   use, and why?
