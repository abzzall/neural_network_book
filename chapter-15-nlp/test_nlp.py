"""Check the from-scratch text tools against definitions and libraries."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


def test_bpe_merges_reduce_symbols_and_round_trip(load_script):
    bpe = load_script('bpe')
    corpus = {'lower': 5, 'lowest': 2, 'newer': 6, 'wider': 3}
    merges, vocab = bpe.learn(corpus, 10)
    counts = [count for _, count in merges]
    assert counts == sorted(counts, reverse=True)       # greedy: never increases here
    for word in corpus:
        pieces = bpe.encode(word, merges)
        assert ''.join(pieces) == word + bpe.END       # segmentation loses nothing
        assert pieces in vocab
    assert bpe.encode('xyz', merges) == ('x', 'y', 'z', bpe.END)   # unseen letters survive


def test_tfidf_matches_scikit_learn(load_script):
    tfidf = load_script('tfidf_numpy')
    docs = ['the cat sat', 'the cat sat on the mat', 'dogs and cats', 'a mat, a cat!']
    vocab, _, x = tfidf.tfidf(docs)
    vectorizer = TfidfVectorizer()
    reference = vectorizer.fit_transform(docs).toarray()
    assert vocab == list(vectorizer.get_feature_names_out())
    assert np.allclose(x, reference, atol=1e-12)


def test_sgns_gradients_match_central_differences(load_script):
    sgns = load_script('sgns')
    rng = np.random.default_rng(0)
    v, u, negatives = rng.normal(size=4), rng.normal(size=4), rng.normal(size=(3, 4))
    _, gv, gu, gn = sgns.loss_and_grads(v, u, negatives)
    h = 1e-6
    for array, grad in ((v, gv), (u, gu), (negatives, gn)):
        numeric = np.zeros_like(array)
        for idx in np.ndindex(array.shape):
            old = array[idx]
            array[idx] = old + h
            plus = sgns.loss_and_grads(v, u, negatives)[0]
            array[idx] = old - h
            minus = sgns.loss_and_grads(v, u, negatives)[0]
            array[idx] = old
            numeric[idx] = (plus - minus) / (2 * h)
        assert np.allclose(grad, numeric, atol=1e-7)


def test_repeated_negative_receives_both_updates(load_script):
    sgns = load_script('sgns')
    V, U = np.ones((3, 2)), np.ones((3, 2))
    sgns.sgd_step(V, U, 0, 1, np.array([2, 2]), lr=1.0)
    single = sgns.sigmoid(2.0)                           # u . v = 2 before the step
    assert U[2] == pytest.approx(np.ones(2) - 2 * single)


def test_pairs_and_noise(load_script):
    sgns = load_script('sgns')
    assert list(sgns.pairs([np.array([0, 1, 2])], window=1)) == [
        (0, 1), (1, 0), (1, 2), (2, 1)]
    p = sgns.noise_distribution([np.array([0, 0, 0, 1])], 3)
    assert p[2] == 0 and p.sum() == pytest.approx(1)
    assert p[0] / p[1] == pytest.approx(3 ** 0.75)
