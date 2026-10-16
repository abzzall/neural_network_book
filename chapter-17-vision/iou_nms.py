"""Continuous-coordinate box overlap and stable, single-class suppression."""

import numpy as np


def boxes_array(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('expected finite (N,4) boxes with positive width and height')
    return boxes


def pairwise_iou(first, second):
    first, second = boxes_array(first), boxes_array(second)
    corner = np.maximum(first[:, None, :2], second[None, :, :2])
    end = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(end - corner, 0).prod(axis=-1)
    area_a = (first[:, 2:] - first[:, :2]).prod(axis=-1)
    area_b = (second[:, 2:] - second[:, :2]).prod(axis=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def nms(boxes, scores, threshold=0.5):
    boxes, scores = boxes_array(boxes), np.asarray(scores, dtype=float)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError('one finite score per box is required')
    if not 0 <= threshold <= 1:
        raise ValueError('IoU threshold must be in [0,1]')
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order):
        index = order[0]
        kept.append(int(index))
        remaining = order[1:]
        overlap = pairwise_iou(boxes[index:index + 1], boxes[remaining])[0]
        order = remaining[overlap <= threshold]
    return kept
