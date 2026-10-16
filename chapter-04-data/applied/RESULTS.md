# Chapter 4 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-04-data/applied/full-code/verify_results.py chapter-04-data/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `rows_before` | |
| `district_spellings_before` | |
| `rows_after_cleaning` | |
| `missing_area_after_cleaning` | |
| `median_area_train` | |
| `median_year_train` | |
| `validation_rmse` | |
| `validation_rmse_with_valuation` | |

## What to explain

1. Why must the medians and the scaling be computed after the split, from the
   training rows only? What would go wrong otherwise?
2. The valuation makes the validation error about fourteen times smaller. Why
   must it still be left out of the model?
3. The rule "area above 400 m2 is a mistake" misses some units mistakes. Which
   ones, and how could you find them?
