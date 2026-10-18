# Frozen sentiment-model probe

Environment and exact checkpoint revision:

Save the complete output as `results.json` next to this file.

| Quantity | Recorded value |
| --- | --- |
| threshold | |
| calibration_accuracy (threshold-selection set) | |
| test_accuracy_before | |
| test_accuracy_after | |
| test_errors (list, possibly empty) | |
| shift_accuracy (product and service reviews) | |
| shift_errors | |
| shift_error_scores | |

Explain:

1. Why must test labels not choose the threshold?
2. What changes, and what stays fixed, in this probe?
3. What do the observed errors, or their absence, establish?
4. Why do these results not establish probability calibration or performance on real reviews?
5. Could another threshold have repaired the errors on the product and service
   reviews? What do their scores say about the model's confidence?
6. What data would support a meaningful fine-tuning and evaluation experiment?
