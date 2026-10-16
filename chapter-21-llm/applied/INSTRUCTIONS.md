# Chapter 21 applied exercise — retrieval-augmented generation

## 1. Outcome

A question-answering system for facts about this book's own examples. A small
instruction-tuned model answers 20 fixed questions twice: from the question
alone, and with the two most similar passages of `corpus.md` added to the
prompt. Retrieval uses the sentence encoder of chapter 15. Generation is greedy,
so every run gives the same answers on the same platform.

## 2. Verified versions

Python 3.14.4, PyTorch 2.14.1 (CPU), transformers 5.20.1, pytest 9.1.1,
17 October 2026. The complete solution takes about one minute.

## 3. Models (download once)

```bash
.venv/bin/python chapter-21-llm/lm.py --download qwen
.venv/bin/python chapter-15-nlp/encoder.py --download
```

- `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae55760`, Apache-2.0 licence (about 1 GB).
- `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243`, Apache-2.0 licence.

Both are cached in `data/` and never committed. `corpus.md` and
`questions.json` are original material under this repository's MIT licence.

## 4. Steps

1. `chunks`: split the corpus into passages at blank lines; drop empty pieces.
2. `retrieve`: for each question vector, return the indices of the `k`
   passages with the highest cosine similarity (the vectors are normalised).
3. `prompt`: build the chat prompt; with context, put the passages under the
   heading `Context:` followed by `Question: ...`.
4. `correct`: lower-case the answer, remove commas, and accept it if it contains
   any accepted string.
5. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-21-llm/applied/based-on
.venv/bin/python chapter-21-llm/applied/based-on/mini_rag.py
```

## 5. Verification

```bash
.venv/bin/python chapter-21-llm/applied/full-code/verify_results.py chapter-21-llm/applied/RESULTS.md
```

Counts may differ by one or two across platforms, because a tiny numerical
difference can change a greedy token.
