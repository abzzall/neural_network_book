"""Chapter 8: classification metrics from a confusion matrix, checked with scikit-learn."""

import numpy as np
from sklearn import metrics as skm

from nnbook.metrics import precision_recall_f1, roc_auc

tp, fp, fn, tn = 40, 10, 20, 130                       # a screening test on 200 people
y_true = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn)
y_pred = np.array([1] * tp + [1] * fp + [0] * fn + [0] * tn)

accuracy = np.mean(y_true == y_pred)
precision, recall, f1 = precision_recall_f1(y_true, y_pred)

scores = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1])   # four positives, four negatives
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

if __name__ == "__main__":
    print(f"accuracy = {accuracy:.4f}, precision = {precision:.4f}, "
          f"recall = {recall:.4f}, F1 = {f1:.4f}")
    print("matches scikit-learn:", bool(np.isclose(f1, skm.f1_score(y_true, y_pred))))
    print(f"ROC-AUC = {roc_auc(labels, scores):.4f} "
          f"(scikit-learn {skm.roc_auc_score(labels, scores):.4f})")
