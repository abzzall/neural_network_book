"""Chapter 5: softmax regression on 8x8 images of handwritten digits.

Data: the UCI Optical Recognition of Handwritten Digits test set (Alpaydin and
Kaynak, 1998, CC BY 4.0), as distributed with scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression as SkLogistic

from nnbook import SoftmaxRegression
from nnbook.preprocessing import StandardScaler, train_test_split

X, y = load_digits(return_X_y=True)               # 1797 images of 64 pixels
train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
scaler = StandardScaler().fit(X[train])
X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])

model = SoftmaxRegression(learning_rate=0.5, epochs=500).fit(X_train, y[train])
accuracy = np.mean(model.predict(X_test) == y[test])
reference = SkLogistic(max_iter=2000).fit(X_train, y[train]).score(X_test, y[test])

if __name__ == "__main__":
    print(f"images: {len(y)}, pixels per image: {X.shape[1]}, classes: {len(np.unique(y))}")
    print(f"training loss: first epoch {model.losses[0]:.4f}, last {model.losses[-1]:.4f}")
    print(f"test accuracy, from scratch: {accuracy:.4f}")
    print(f"test accuracy, scikit-learn (L2, C = 1): {reference:.4f}")
