# Chapter 20 applied exercise — results

Fill the Value column with four or more significant figures (means over seeds 0, 1, 2).

| Quantity | Value |
|---|---|
| `test_items` | |
| `test_anomalies` | |
| `ae_recall` | |
| `ae_false_alarm` | |
| `ae_precision` | |
| `ae_precision_at_1pct` | |
| `pca_recall` | |
| `pca_false_alarm` | |
| `pca_precision` | |
| `pca_precision_at_1pct` | |

## What to explain

1. The autoencoder's AUC for digit 3 is about 0.85, yet it catches only about
   a third of the anomalies. How can both be true?
2. Why does precision fall so sharply when anomalies become rare? What does
   that mean for a factory that inspects thousands of parts a day?
3. How would you choose the threshold if a missed defect cost 100 times more
   than a false alarm?
4. Why must the threshold be set on validation data and not on the test set?
