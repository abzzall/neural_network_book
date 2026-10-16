"""nnbook: neural networks from scratch, the companion library of the textbook Neural Networks."""

from nnbook import preprocessing
from nnbook.linear import LogisticRegression, SoftmaxRegression
from nnbook.perceptron import Perceptron

__all__ = ["LogisticRegression", "Perceptron", "SoftmaxRegression", "preprocessing"]
__version__ = "0.1.0"
