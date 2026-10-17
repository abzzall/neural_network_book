"""How well do the other features reveal age? Predict 'aged 25 or less' without age."""

from sklearn.model_selection import train_test_split

from credit_data import load
from credit_model import model
from permutation_importance import permutation_importance

X, _, young = load()
X = X.drop(columns='age')
X_tr, X_te, y_tr, y_te = train_test_split(X, young, test_size=0.3, random_state=0,
                                          stratify=young)
fitted = model(X.columns).fit(X_tr, y_tr)
auc, drops = permutation_importance(fitted, X_te, y_te, repeats=20)
print(f'AUC for predicting the young group from the other 19 features: {auc:.3f}')
for column in sorted(drops, key=drops.get, reverse=True)[:3]:
    print(f'  {column}: importance {drops[column]:.3f}')
