"""TF-IDF from scratch, checked against scikit-learn's TfidfVectorizer."""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):
    return re.findall(r'(?u)\b\w\w+\b', text.lower())   # scikit-learn's default rule


def tfidf(docs):
    tokens = [tokenize(doc) for doc in docs]
    vocab = sorted({token for doc in tokens for token in doc})
    index = {token: j for j, token in enumerate(vocab)}
    counts = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(tokens):
        for token in doc:
            counts[i, index[token]] += 1
    df = (counts > 0).sum(axis=0)                      # documents containing each term
    idf = np.log((1 + len(docs)) / (1 + df)) + 1       # smoothed inverse document frequency
    weights = counts * idf
    return vocab, idf, weights / np.linalg.norm(weights, axis=1, keepdims=True)


if __name__ == '__main__':
    docs = ['Win a free prize now', 'Call me when you are free', 'Free call now to win']
    vocab, idf, x = tfidf(docs)
    print('vocabulary:', ' '.join(vocab))
    print('idf:', ' '.join(f'{v:.4f}' for v in idf))
    print('document 1:', ' '.join(f'{v:.4f}' for v in x[0]))
    reference = TfidfVectorizer().fit_transform(docs).toarray()
    print(f'max difference from scikit-learn: {np.abs(x - reference).max():.1e}')
    print('cosine(doc 1, doc 3):', f'{x[0] @ x[2]:.4f}')
