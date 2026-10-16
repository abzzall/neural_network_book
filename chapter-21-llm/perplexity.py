"""Perplexity of GPT-2 on a sentence and on the same words in random order."""

import math

import numpy as np
import torch

from lm import load


def perplexity(tokenizer, model, text):
    """exp of the mean next-token cross-entropy, eq. (21.1)."""
    ids = tokenizer(text, return_tensors='pt').input_ids
    with torch.inference_mode():
        loss = model(ids, labels=ids).loss          # mean over predicted tokens
    return math.exp(loss.item()), ids.shape[1]


if __name__ == '__main__':
    torch.set_num_threads(4)
    tokenizer, model = load('gpt2')
    text = ('The students opened their books and began to read the first chapter '
            'about neural networks.')
    words = text.rstrip('.').split()
    shuffled = ' '.join(np.random.default_rng(0).permutation(words)) + '.'
    for name, t in (('original', text), ('shuffled', shuffled)):
        ppl, n = perplexity(tokenizer, model, t)
        print(f'{name}: {n} tokens, perplexity {ppl:.1f}')
    print('shuffled text:', shuffled)
