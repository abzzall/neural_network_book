"""nnbook: neural networks from scratch, the companion library of the textbook Neural Networks."""

from nnbook import preprocessing
from nnbook.activations import ReLU, Sigmoid, Tanh
from nnbook.gradcheck import max_relative_error
from nnbook.layers import Dense
from nnbook.linear import LogisticRegression, SoftmaxRegression
from nnbook.losses import MeanSquaredError, SoftmaxCrossEntropy
from nnbook.network import Network
from nnbook.perceptron import Perceptron

__all__ = ["Dense", "LogisticRegression", "MeanSquaredError", "Network", "Perceptron",
           "ReLU", "Sigmoid", "SoftmaxCrossEntropy", "SoftmaxRegression", "Tanh",
           "max_relative_error", "preprocessing"]
__version__ = "0.1.0"
