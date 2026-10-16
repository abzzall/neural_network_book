"""nnbook: neural networks from scratch, the companion library of the textbook Neural Networks."""

from nnbook import metrics, model_selection, preprocessing
from nnbook.activations import ReLU, Sigmoid, Tanh
from nnbook.gradcheck import max_relative_error
from nnbook.layers import Dense
from nnbook.linear import LogisticRegression, SoftmaxRegression
from nnbook.losses import MeanSquaredError, SoftmaxCrossEntropy
from nnbook.network import Network
from nnbook.optimizers import SGD, AdaGrad, Adam, RMSProp
from nnbook.perceptron import Perceptron
from nnbook.regularization import BatchNorm, Dropout

__all__ = ["AdaGrad", "Adam", "BatchNorm", "Dense", "Dropout", "RMSProp", "SGD", "LogisticRegression", "MeanSquaredError", "Network", "Perceptron",
           "ReLU", "Sigmoid", "SoftmaxCrossEntropy", "SoftmaxRegression", "Tanh",
           "max_relative_error", "metrics", "model_selection", "preprocessing"]
__version__ = "0.1.0"
