"""Complete the three segmentation functions, then run the fixed experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from segment_shapes import run

def class_dice(actual, predicted):
    actual, predicted = actual.bool(), predicted.bool()
    scores = []
    for category in (False, True):
        a, p = actual == category, predicted == category
        denominator = a.sum() + p.sum()
        scores.append(float(2 * (a & p).sum() / denominator)
                      if denominator else 1.)
    return scores


def segmentation_loss(logits, masks):
    probabilities = logits.sigmoid()
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = (probabilities + masks).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return F.binary_cross_entropy_with_logits(logits, masks) + .5 * (1 - dice.mean())


def choose_threshold(probabilities, masks):
    candidates = (.2, .3, .4, .5, .6, .7, .8)
    return max(candidates, key=lambda t: (
        sum(class_dice(masks, probabilities >= t)), -abs(t - .5), -t))


if __name__ == '__main__':
    import json
    print(json.dumps([run(s, loss_fn=segmentation_loss,
                         threshold_fn=choose_threshold, score_fn=class_dice)
                      for s in (42, 43, 44)], indent=2))
