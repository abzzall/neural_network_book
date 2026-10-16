# Chapter 15 — Working with Text

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4, scikit-learn
1.9.1, PyTorch 2.14.1 and transformers 5.20.1 (16 October 2026). Use the
environment described in chapter 9; transformers is installed with chapter 14.
Run commands from the repository root.

| File | What is checked |
|---|---|
| `normalise_demo.py` | Code points, UTF-8 byte counts, NFC/NFKC normalisation and case folding (a Kazakh word) |
| `bpe.py` | Byte-pair encoding: merges learned from word counts; segmentation of unseen words |
| `tfidf_numpy.py` | TF-IDF from scratch; identical to scikit-learn's `TfidfVectorizer` |
| `sgns.py`, `skipgram_demo.py` | Skip-gram with negative sampling: gradients against central differences; word neighbours on a generated corpus |
| `spam_data.py` | Download (hash-checked), loading and deduplicated stratified split of the SMS Spam Collection |
| `tfidf_baseline.py` | TF-IDF + logistic regression on the deduplicated split |
| `duplicates_demo.py` | How duplicated messages inflate test scores, over ten splits |
| `encoder.py`, `semantic_search.py` | Sentence embeddings from a pinned pretrained encoder; search by meaning vs by words |
| `applied/` | 200 labels: TF-IDF vs frozen encoder vs fine-tuned encoder |

The first five rows need no download. The rest need the data and the model:

```bash
.venv/bin/python chapter-15-nlp/spam_data.py --download
.venv/bin/python chapter-15-nlp/encoder.py --download
```

Tests that need them are skipped until they are downloaded.

```bash
.venv/bin/python chapter-15-nlp/bpe.py
.venv/bin/python chapter-15-nlp/tfidf_numpy.py
.venv/bin/python chapter-15-nlp/skipgram_demo.py
.venv/bin/python chapter-15-nlp/tfidf_baseline.py
.venv/bin/python chapter-15-nlp/duplicates_demo.py
.venv/bin/python chapter-15-nlp/semantic_search.py
.venv/bin/python -m pytest chapter-15-nlp --ignore-glob='*/based-on/*'
```

Expected output of `bpe.py`:

```text
merge 1: a + i (count 15)
merge 2: ai + n (count 15)
merge 3: r + ain (count 15)
merge 4: rain + </w> (count 13)
merge 5: t + rain</w> (count 4)
merge 6: b + rain</w> (count 3)
rains -> rain s </w>
strain -> s train</w>
grain -> g rain</w>
```

Expected output of `tfidf_baseline.py` (last digits can vary across platforms):

```text
test messages: 1293 (spam 163)
true ham 1114, false spam 16, missed spam 12, caught spam 151
spam precision: 0.9042, spam recall: 0.9264
macro-F1: 0.9514
strongest spam words: call, txt, text, reply, free, stop, mobile, www
strongest ham words: my, me, that, gt, lt, ok, but, at
messages containing &lt;: 222, of them spam: 0
```

`gt` and `lt` are the remains of the HTML entities `&gt;` and `&lt;` in the raw
messages. They occur in 222 distinct messages, none of them spam, so the model
has learned a formatting artefact of the collection: a reminder to inspect
data before trusting a model's features.

## Data and model

- **SMS Spam Collection v.1** (5574 messages, 747 spam), T. A. Almeida and
  J. M. Gomez Hidalgo, UCI Machine Learning Repository,
  doi 10.24432/C5CC84. Provided free of charge with no limitations other than
  the authors' copyright and a warranty disclaimer; please cite Almeida,
  Gomez Hidalgo and Yamakami (2011). Downloaded by `spam_data.py` and checked
  against a SHA-256 hash; not stored in this repository.
- **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), Apache-2.0 licence; mean
  pooling as in its model card. Cached in `data/`; not stored in this repository.
- The generated corpus in `skipgram_demo.py` and the sentences in
  `semantic_search.py` are original material under this repository's MIT licence.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`. The incomplete `based-on/` functions
raise errors naming the step to complete; the finished `full-code/` solution
passes the same acceptance tests.

```bash
.venv/bin/python chapter-15-nlp/applied/full-code/spam_compare.py
.venv/bin/python chapter-15-nlp/applied/full-code/verify_results.py
```

Reference output (last digits can vary across platforms):

```text
labelled_messages: 200
labelled_spam: 25
test_messages: 1293
tfidf_macro_f1: 0.888803
frozen_macro_f1: 0.908567
finetuned_macro_f1_mean: 0.964973
finetuned_macro_f1_std: 0.00217104
```
