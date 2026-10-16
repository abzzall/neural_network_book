"""Binary segmentation losses and globally pooled class Dice scores."""

import torch
from torch.nn import functional as F


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
