# Chapter 21 — Large Language Models

Examples verified on a CPU with Python 3.14.4, PyTorch 2.14.1 and transformers
5.20.1 (17 October 2026). Run commands from the repository root. No library
beyond `requirements.txt` is needed.

| File | What is checked |
|---|---|
| `decoding.py` | Temperature, top-k and top-p from scratch on a six-token distribution |
| `compute_budget.py` | Compute-optimal model size from the fitted loss of Hoffmann et al. (2022) |
| `dpo_toy.py` | Direct preference optimisation recovers the KL-regularised optimum |
| `lm.py` | Pinned GPT-2 and Qwen2.5-0.5B-Instruct; downloads only on request |
| `perplexity.py` | GPT-2 perplexity of a sentence and of its shuffled words |
| `kv_cache_demo.py` | Greedy generation with and without the key-value cache |
| `applied/` | Retrieval-augmented question answering over facts about the book |

```bash
.venv/bin/python chapter-21-llm/decoding.py
.venv/bin/python chapter-21-llm/compute_budget.py
.venv/bin/python chapter-21-llm/dpo_toy.py
.venv/bin/python chapter-21-llm/lm.py --download gpt2 qwen
.venv/bin/python chapter-21-llm/perplexity.py
.venv/bin/python chapter-21-llm/kv_cache_demo.py
.venv/bin/python -m pytest chapter-21-llm --ignore-glob='*/based-on/*'
```

Expected output of `decoding.py`:

```text
probabilities:     0.4000 0.2500 0.1500 0.1000 0.0600 0.0400
temperature 0.5:   0.6149 0.2402 0.0865 0.0384 0.0138 0.0061
temperature 2.0:   0.2773 0.2192 0.1698 0.1386 0.1074 0.0877
top-k, k = 3:      0.5000 0.3125 0.1875 0.0000 0.0000 0.0000
top-p, p = 0.75:   0.5000 0.3125 0.1875 0.0000 0.0000 0.0000
top-p, p = 0.60:   0.6154 0.3846 0.0000 0.0000 0.0000 0.0000
share of 10000 top-p draws per token: 0.4990 0.3122 0.1888 0.0000 0.0000 0.0000
```

## Models

- **GPT-2** (`openai-community/gpt2`, revision `607a30d7`), MIT licence, 124M parameters.
- **Qwen2.5-0.5B-Instruct** (`Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae55760`),
  Apache-2.0 licence.

Both are cached in `data/` and never committed; tests that need them are
skipped until they are downloaded.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`.

```bash
.venv/bin/python chapter-21-llm/applied/full-code/mini_rag.py
.venv/bin/python chapter-21-llm/applied/full-code/verify_results.py
```

Reference output:

```text
passages: 20
questions: 20
retrieval_hits: 20
correct_without_retrieval: 1
correct_with_retrieval: 19
```
