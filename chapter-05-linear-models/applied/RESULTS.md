# Chapter 5 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-05-linear-models/applied/full-code/verify_results.py chapter-05-linear-models/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `final_training_loss` | |
| `test_accuracy` | |
| `missed_malignant` | |
| `false_alarms` | |
| `sklearn_test_accuracy` | |
| `largest_weight_feature` | |
| `largest_weight` | |

## What to explain

1. All five largest weights are positive. What does a positive weight on a
   standardised feature mean for the predicted probability of malignancy?
2. The model misses some malignant tumours but raises no false alarms. Which
   error is worse in this setting, and how could you change the decision
   threshold of 0.5 to reduce it? What would that cost?
3. Why can the weights be compared with each other only because the features
   were standardised?
