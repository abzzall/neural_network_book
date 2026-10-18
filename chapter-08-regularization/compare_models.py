"""Chapter 8: is a network really better than logistic regression on the digits?"""

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
          "network": make_pipeline(StandardScaler(), MLPClassifier(
              hidden_layer_sizes=(64,), max_iter=2000, random_state=0))}

# 1. The same ten folds for both models: the scores are paired.
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = {name: cross_val_score(m, X, y, cv=folds) for name, m in models.items()}
d = scores["network"] - scores["logistic"]               # one difference per fold
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
p_value = 2 * stats.t.sf(abs(t), df=len(d) - 1)

# 2. One test set of 450 digits: bootstrap the paired difference in accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=450, stratify=y,
                                                    random_state=0)
right = {name: m.fit(X_train, y_train).predict(X_test) == y_test
         for name, m in models.items()}
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    i = rng.integers(0, len(y_test), len(y_test))      # resample test digits
    boot.append(right["network"][i].mean() - right["logistic"][i].mean())
low, high = np.percentile(boot, [2.5, 97.5])

if __name__ == "__main__":
    for name, s in scores.items():
        print(f"{name:8s} 10-fold accuracy = {s.mean():.4f} +- {s.std():.4f}")
    print(f"paired differences: mean = {d.mean():.4f}, sd = {d.std(ddof=1):.4f}, "
          f"network better in {np.sum(d > 0)} folds, worse in {np.sum(d < 0)}")
    print(f"paired t = {t:.3f}, p = {p_value:.3f} "
          f"(scipy: {stats.ttest_rel(scores['network'], scores['logistic']).pvalue:.3f})")
    only_net = np.sum(right["network"] & ~right["logistic"])
    only_log = np.sum(right["logistic"] & ~right["network"])
    print(f"test set: difference = {right['network'].mean() - right['logistic'].mean():.4f}, "
          f"95% bootstrap interval [{low:.4f}, {high:.4f}]")
    print(f"digits only the network gets right: {only_net}, only logistic: {only_log}")
