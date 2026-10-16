"""Chapter 5 applied exercise (starting point): logistic regression from scratch on tumour measurements.

Data: Breast Cancer Wisconsin (Diagnostic), Wolberg, Mangasarian, Street and
Street (UCI, CC BY 4.0), as distributed with scikit-learn. This is a teaching
exercise, not a diagnostic tool.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SkLogistic

from nnbook import LogisticRegression
from nnbook.preprocessing import StandardScaler, train_test_split


# Step 1: load the data; scikit-learn codes benign as 1, so flip to make malignant = 1.
def load():
    data = load_breast_cancer()
    return data.data, 1 - data.target, list(data.feature_names)


# Step 2: a stratified 75/25 split and standardisation fitted on the training rows.
def prepare(X, y, seed=0):
    raise NotImplementedError("Step 2: stratified 75/25 split (seed 0), then standardise with training statistics")


# Step 3: train logistic regression by gradient descent.
def train(X_train, y_train, learning_rate=0.1, epochs=2000):
    raise NotImplementedError("Step 3: nnbook.LogisticRegression(learning_rate, epochs).fit(...)")


# Step 4: accuracy, and the two kinds of error counted separately.
def evaluate(model, X_test, y_test):
    raise NotImplementedError("Step 4: accuracy, missed malignant tumours, false alarms")


# Step 5: the reference implementation with its default settings.
def reference_accuracy(X_train, X_test, y_train, y_test):
    raise NotImplementedError("Step 5: scikit-learn LogisticRegression(max_iter=5000) test accuracy")


# Step 6: the features with the largest weights, and their signs.
def largest_weights(model, names, k=5):
    raise NotImplementedError("Step 6: the k features with the largest absolute weights, with their weights")
