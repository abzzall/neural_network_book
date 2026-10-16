"""Evaluation metrics (chapter 8)."""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes=None):
    """Entry (i, j) counts samples of true class i predicted as class j."""
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def precision_recall_f1(y_true, y_pred):
    """For two classes, with class 1 as the positive class."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores):
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def mse(y, y_hat):
    return float(np.mean((y - y_hat) ** 2))


def mae(y, y_hat):
    return float(np.mean(np.abs(y - y_hat)))


def r2(y, y_hat):
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
