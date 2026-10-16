"""Greedy generation with and without the key-value cache: same text, different cost."""

import time

import torch

from lm import load


def cache_bytes(config, tokens, bytes_per_value=4):
    """Memory of the cache, eq. (21.4): keys and values for every layer and position."""
    return 2 * config.n_layer * tokens * config.n_embd * bytes_per_value


if __name__ == '__main__':
    torch.set_num_threads(4)
    tokenizer, model = load('gpt2')
    prompt = tokenizer('A neural network learns by', return_tensors='pt')
    outputs = {}
    for use_cache in (True, False):
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**prompt, max_new_tokens=60, do_sample=False,
                                 use_cache=use_cache, pad_token_id=tokenizer.eos_token_id)
        outputs[use_cache] = out
        print(f'use_cache={use_cache}: {time.perf_counter() - start:.2f} s')
    print('identical tokens:', torch.equal(outputs[True], outputs[False]))
    n = outputs[True].shape[1]
    print(f'cache for {n} tokens: {cache_bytes(model.config, n) / 1e6:.2f} MB '
          f'({model.config.n_layer} layers, width {model.config.n_embd})')
    print('text:', tokenizer.decode(outputs[True][0]).replace('\n', ' '))
