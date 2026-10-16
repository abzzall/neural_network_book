"""Run every chapter 5 script as a reader would and check what it prints."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script):
    result = subprocess.run([sys.executable, str(HERE / script)],
                            capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_perceptron_trace():
    assert run("perceptron_trace.py") == [
        "epoch 1: x = [0 0], y = -1 -> w = [0. 0.], b = -1.0",
        "epoch 1: x = [1 1], y = +1 -> w = [1. 1.], b = 0.0",
        "epoch 2: x = [0 0], y = -1 -> w = [1. 1.], b = -1.0",
        "epoch 2: x = [0 1], y = -1 -> w = [1. 0.], b = -2.0",
        "epoch 2: x = [1 1], y = +1 -> w = [2. 1.], b = -1.0",
        "epoch 9: no mistakes; final w = [3. 2.], b = -4.0",
    ]


def test_xor_demo():
    assert run("xor_demo.py") == [
        "perceptron mistakes in epoch 100: 4",
        "logistic regression probabilities: [0.5 0.5 0.5 0.5]",
        "with the feature x1*x2:           [0.015 0.99  0.99  0.007]",
    ]


def test_softmax_digits():
    assert run("softmax_digits.py") == [
        "images: 1797, pixels per image: 64, classes: 10",
        "training loss: first epoch 2.3026, last 0.0429",
        "test accuracy, from scratch: 0.9556",
        "test accuracy, scikit-learn (L2, C = 1): 0.9600",
    ]
