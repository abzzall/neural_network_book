"""One image, one class: greedy matching and all-points interpolated AP."""

import numpy as np

from iou_nms import boxes_array, pairwise_iou


def average_precision(boxes, scores, truth, threshold=0.5):
    boxes, truth = boxes_array(boxes), boxes_array(truth)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError('one finite score per prediction is required')
    if not 0 <= threshold <= 1:
        raise ValueError('matching threshold must be in [0,1]')
    order = np.argsort(-scores, kind='stable')
    overlap = pairwise_iou(boxes[order], truth)
    matched = np.zeros(len(truth), dtype=bool)
    true_positive = np.zeros(len(boxes), dtype=float)
    for rank, row in enumerate(overlap):
        available = np.where(matched, -1., row)
        if len(truth) and available.max() >= threshold:
            match = available.argmax()
            matched[match] = True
            true_positive[rank] = 1.
    cumulative = true_positive.cumsum()
    precision = cumulative / np.arange(1, len(boxes) + 1)
    recall = cumulative / len(truth) if len(truth) else cumulative * 0
    r = np.concatenate(([0.], recall, [1.]))
    p = np.concatenate(([0.], precision, [0.]))
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    ap = np.sum(np.diff(r) * envelope[1:]) if len(truth) else 0.
    return float(ap), true_positive.astype(int), precision, recall
