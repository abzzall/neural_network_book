"""Chapter 25: a regression tree split by hand, and gradient boosting from scratch."""

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def best_split(x, y):
    """The threshold on one feature that minimises the summed squared error of two leaves."""
    best = (np.inf, None)
    for t in (np.sort(x)[:-1] + np.sort(x)[1:]) / 2:    # midpoints between sorted values
        left, right = y[x <= t], y[x > t]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        best = min(best, (sse, t))
    return best


def boost(X, y, trees=200, rate=0.1, depth=3):
    """Start from the mean; each small tree fits the residuals of the current model."""
    prediction, ensemble = np.full(len(y), y.mean()), []
    rng = np.random.RandomState(0)          # breaks ties between equal splits, as sklearn does
    for _ in range(trees):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=rng).fit(X, y - prediction)
        prediction += rate * tree.predict(X)
        ensemble.append(tree)
    return y.mean(), ensemble


def predict(model, X, rate=0.1, trees=None):
    start, ensemble = model
    return start + rate * sum(t.predict(X) for t in ensemble[:trees])


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])         # worked example
    y = np.array([2.0, 3.0, 2.5, 7.0, 8.0, 7.5])
    sse, t = best_split(x, y)
    print(f"best split x <= {t}: leaf means {y[x <= t].mean():.2f} and "
          f"{y[x > t].mean():.2f}, squared error {sse:.2f} (no split: "
          f"{((y - y.mean()) ** 2).sum():.2f})")
    X, Y = make_friedman1(n_samples=1500, noise=1.0, random_state=0)
    model = boost(X[:1000], Y[:1000])
    reference = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3,
                                          random_state=0)
    reference.fit(X[:1000], Y[:1000])
    gap = np.abs(predict(model, X[1000:]) - reference.predict(X[1000:])).max()
    print(f"largest difference from scikit-learn: {gap:.2e}")
    for m in (1, 10, 50, 200):
        mse = lambda rows: np.mean((predict(model, X[rows], trees=m) - Y[rows]) ** 2)
        print(f"{m:3d} trees: training MSE {mse(slice(0, 1000)):.3f}, "
              f"validation MSE {mse(slice(1000, 1500)):.3f}")
