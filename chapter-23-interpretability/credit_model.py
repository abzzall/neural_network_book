"""A logistic-regression credit model with one-hot encoded categories."""

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_data import load


def model(columns):
    categorical = [c for c in columns if c.startswith(('checking', 'history', 'purpose',
        'savings', 'employment', 'status', 'guarantors', 'property', 'other', 'housing',
        'job', 'telephone', 'foreign'))]
    numeric = [c for c in columns if c not in categorical]
    prepare = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categorical),
                                      (StandardScaler(), numeric))
    return make_pipeline(prepare, LogisticRegression(C=1.0, max_iter=2000))


def data(drop=()):
    X, good, young = load()
    X = X.drop(columns=list(drop))
    return train_test_split(X, good, young, test_size=0.3, random_state=0, stratify=good)
