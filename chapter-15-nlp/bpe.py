"""Byte-pair encoding: learn merges from word counts, then segment new words."""

from collections import Counter

END = '</w>'   # end-of-word marker, so "in" inside a word differs from a final "in"


def pair_counts(vocab):
    counts = Counter()
    for symbols, freq in vocab.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def learn(word_counts, n_merges):
    vocab = {tuple(word) + (END,): freq for word, freq in word_counts.items()}
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(vocab)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))  # most frequent; ties alphabetical
        merges.append((best, counts[best]))
        vocab = {merge(symbols, best): freq for symbols, freq in vocab.items()}
    return merges, vocab


def encode(word, merges):
    symbols = tuple(word) + (END,)
    for pair, _ in merges:
        symbols = merge(symbols, pair)
    return symbols


if __name__ == '__main__':
    corpus = {'rain': 5, 'train': 4, 'brain': 3, 'trains': 2, 'drain': 1}
    merges, vocab = learn(corpus, 6)
    for k, ((left, right), count) in enumerate(merges, start=1):
        print(f'merge {k}: {left} + {right} (count {count})')
    for word in ('rains', 'strain', 'grain'):
        print(f'{word} -> {" ".join(encode(word, merges))}')
