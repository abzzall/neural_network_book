# Chapter 8 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-08-regularization/applied/full-code/verify_results.py chapter-08-regularization/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `baseline_val_accuracy` | |
| `l2_val_accuracy` | |
| `dropout_val_accuracy` | |
| `early_stopping_val_accuracy` | |
| `batch_norm_val_accuracy` | |
| `early_stopping_epochs` | |
| `chosen_setting` | |
| `chosen_test_accuracy` | |

## What to explain

1. Every setting reaches 100 per cent training accuracy. Why is the training
   accuracy useless for choosing between them?
2. Why is only the chosen setting evaluated on the test set?
3. Early stopping does not give the best validation accuracy here. What does it
   give instead, and when would you still use it?
