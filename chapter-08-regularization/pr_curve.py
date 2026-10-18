"""Precision-recall and average precision on a rare-event ranking."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def ranked_curve(labels, scores):
    """Distinct scores: cumulative precision and recall, descending by score."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    positives = np.asarray(labels)[order]
    tp = np.cumsum(positives)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / positives.sum()
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    return precision, recall, float(ap)


if __name__ == "__main__":
    labels = np.zeros(100, dtype=int)
    labels[[0, 2, 9, 19, 49]] = 1
    scores = np.linspace(1, 0, len(labels))
    precision, recall, ap = ranked_curve(labels, scores)
    print(f"prevalence {labels.mean():.2f}, average precision {ap:.4f}")
    for rank in (1, 3, 10, 20, 50, 100):
        print(f"top {rank:3}: precision {precision[rank-1]:.4f}, recall {recall[rank-1]:.4f}")
    print(f"matches scikit-learn: {np.isclose(ap, average_precision_score(labels, scores))}")
