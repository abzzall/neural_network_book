"""Four predictions show why duplicate suppression and accuracy differ."""

import numpy as np

from average_precision import average_precision
from iou_nms import nms, pairwise_iou


if __name__ == '__main__':
    truth = np.array([[0, 0, 2, 2], [3, 0, 5, 2]])
    boxes = np.array([[0, 0, 2, 2], [.2, 0, 2.2, 2], [6, 0, 8, 2], [3, 0, 5, 2]])
    scores = np.array([.9, .8, .7, .6])
    kept = nms(boxes, scores, threshold=.5)
    before, flags, precision, recall = average_precision(boxes, scores, truth)
    after, _, _, _ = average_precision(boxes[kept], scores[kept], truth)
    print(f'duplicate IoU: {pairwise_iou(boxes[:1], boxes[1:2])[0, 0]:.6f}')
    print(f'kept prediction indices: {kept}')
    print(f'true-positive flags before NMS: {flags.tolist()}')
    print('precision before NMS: ' + ', '.join(f'{x:.6f}' for x in precision))
    print('recall before NMS: ' + ', '.join(f'{x:.6f}' for x in recall))
    print(f'AP at IoU 0.5 before NMS: {before:.6f}')
    print(f'AP at IoU 0.5 after NMS: {after:.6f}')
