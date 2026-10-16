# Chapter 9 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-09-pytorch/applied/full-code/verify_results.py chapter-09-pytorch/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `seed0_val_accuracy` | |
| `seed0_test_accuracy` | |
| `mean_val_accuracy` | |
| `std_val_accuracy` | |

## What to explain

1. The PyTorch network and the nnbook network of chapter 8 have the same
   architecture and training settings. Why do their accuracies differ?
2. How large must a difference between two models be before you believe it,
   given the spread across seeds you measured?
3. What would go wrong if `accuracy` did not call `model.eval()`?
