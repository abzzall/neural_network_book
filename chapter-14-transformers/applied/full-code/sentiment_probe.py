"""Select a decision threshold on calibration labels, then inspect a test set."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER))
from pretrained_probe import load_classifier, positive_scores


def choose_threshold(probabilities, labels):
    candidates = [0.2, 0.35, 0.5, 0.65, 0.8]
    scores = [np.mean((np.asarray(probabilities) >= t) == labels) for t in candidates]
    best = min(range(len(candidates)),
               key=lambda i: (-scores[i], abs(candidates[i] - 0.5), candidates[i]))
    return candidates[best]


def accuracy(probabilities, labels, threshold):
    return float(np.mean((np.asarray(probabilities) >= threshold) == labels))


def errors(texts, probabilities, labels, threshold):
    return [text for text, probability, label in zip(texts, probabilities, labels)
            if int(probability >= threshold) != label]


def run():
    torch.set_num_threads(1)
    examples = json.loads((CHAPTER / 'applied' / 'reviews.json').read_text())
    classifier = load_classifier()
    calibration_texts, calibration_y = zip(*examples['calibration'])
    test_texts, test_y = zip(*examples['test'])
    calibration_p = positive_scores(classifier, list(calibration_texts))
    threshold = choose_threshold(calibration_p, np.asarray(calibration_y))
    # Test labels do not select the threshold or any model parameters.
    test_p = positive_scores(classifier, list(test_texts))
    return {'threshold': threshold,
            'calibration_accuracy': accuracy(calibration_p, calibration_y, threshold),
            'test_accuracy_before': accuracy(test_p, test_y, 0.5),
            'test_accuracy_after': accuracy(test_p, test_y, threshold),
            'test_errors': errors(test_texts, test_p, test_y, threshold)}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
