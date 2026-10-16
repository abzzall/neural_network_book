# Chapter 3 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-03-probability/applied/full-code/verify_results.py chapter-03-probability/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `clicks` | |
| `p_mle` | |
| `p_gd` | |
| `min_cross_entropy_nats` | |
| `entropy_of_p_mle_nats` | |
| `mu_mle` | |
| `var_mle` | |
| `var_unbiased` | |
| `mu_gd` | |
| `var_gd` | |

## What to explain

1. Why is the smallest cross-entropy equal to the entropy of the estimated
   click probability? What would the cross-entropy be for any other `p`?
2. The maximum-likelihood variance divides by n, the sample variance by n - 1.
   Which is larger here, by how much, and why does the difference vanish as n grows?
3. Gradient descent on the Gaussian works with s = log(variance) instead of the
   variance itself. What could go wrong without this change?
