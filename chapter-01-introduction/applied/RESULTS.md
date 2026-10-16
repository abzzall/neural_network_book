# Chapter 1 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-01-introduction/applied/full-code/verify_results.py chapter-01-introduction/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `irrigation_days_bias_0.5` | |
| `irrigation_days_bias_1.0` | |
| `irrigation_days_bias_1.5` | |
| `flip_bias_day_1` | |
| `flip_bias_day_8` | |

## What to explain

1. Why can raising the bias never reduce the number of irrigation days?
2. Day 8 needs a much larger bias than day 1 before it is irrigated. Which of its
   readings cause this, and through which weights?
3. In chapter 1 the weights were chosen by hand. What would a learning
   algorithm need in order to choose them instead?
