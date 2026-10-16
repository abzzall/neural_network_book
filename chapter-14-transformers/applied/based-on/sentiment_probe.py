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
    raise NotImplementedError('Steps 1-3: score candidates and select a threshold')


def accuracy(probabilities, labels, threshold):
    raise NotImplementedError('Step 4: compute classification accuracy')


def errors(texts, probabilities, labels, threshold):
    raise NotImplementedError('Step 5: collect misclassified texts')


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
