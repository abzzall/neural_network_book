"""Chapter 5 applied exercise: logistic regression from scratch on tumour measurements.

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
    train, test = train_test_split(len(y), test_size=0.25, seed=seed, stratify=y)
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X[train]), scaler.transform(X[test]), y[train], y[test]


# Step 3: train logistic regression by gradient descent.
def train(X_train, y_train, learning_rate=0.1, epochs=2000):
    return LogisticRegression(learning_rate, epochs).fit(X_train, y_train)


# Step 4: accuracy, and the two kinds of error counted separately.
def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    return {"accuracy": float(np.mean(predicted == y_test)),
            "missed_malignant": int(np.sum((predicted == 0) & (y_test == 1))),
            "false_alarms": int(np.sum((predicted == 1) & (y_test == 0)))}


# Step 5: the reference implementation with its default settings.
def reference_accuracy(X_train, X_test, y_train, y_test):
    return float(SkLogistic(max_iter=5000).fit(X_train, y_train).score(X_test, y_test))


# Step 6: the features with the largest weights, and their signs.
def largest_weights(model, names, k=5):
    order = np.argsort(-np.abs(model.w))[:k]
    return [(str(names[j]), float(model.w[j])) for j in order]
