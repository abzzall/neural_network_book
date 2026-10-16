"""TF-IDF + logistic regression: the baseline every text model must beat."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from spam_data import load, split


def baseline():
    return make_pipeline(TfidfVectorizer(),
                         LogisticRegression(class_weight='balanced', max_iter=1000))


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = split(*load())
    model = baseline().fit(train_x, train_y)
    predicted = model.predict(test_x)
    (tn, fp), (fn, tp) = confusion_matrix(test_y, predicted)
    print(f'test messages: {len(test_y)} (spam {sum(test_y)})')
    print(f'true ham {tn}, false spam {fp}, missed spam {fn}, caught spam {tp}')
    print(f'spam precision: {tp / (tp + fp):.4f}, spam recall: {tp / (tp + fn):.4f}')
    print(f'macro-F1: {f1_score(test_y, predicted, average="macro"):.4f}')
    vocab = model[0].get_feature_names_out()
    order = np.argsort(model[1].coef_[0])
    print('strongest spam words:', ', '.join(vocab[order[::-1][:8]]))
    print('strongest ham words:', ', '.join(vocab[order[:8]]))
    markup = [y for x, y in zip(train_x + test_x, train_y + test_y) if '&lt;' in x]
    print(f'messages containing &lt;: {len(markup)}, of them spam: {sum(markup)}')
