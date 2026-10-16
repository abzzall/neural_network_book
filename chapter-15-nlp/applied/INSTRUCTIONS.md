# Chapter 15 applied exercise — spam detection with 200 labels

## 1. Outcome

Three classifiers learn from the same 200 labelled SMS messages and are scored
on the same deduplicated test set of 1293 messages:

1. TF-IDF features with logistic regression (no pretraining);
2. frozen sentence embeddings from a pretrained encoder with logistic regression;
3. the same encoder fine-tuned end to end with a two-class head (three seeds).

The question is how much a pretrained model helps when labels are scarce, and
what that help costs.

## 2. Verified versions

Python 3.14.4, PyTorch 2.14.1 (CPU), transformers 5.20.1, scikit-learn 1.9.1,
pytest 9.1.1, 16 October 2026. The complete solution takes about 80 seconds on
a laptop CPU.

## 3. Data and model (download once)

```bash
.venv/bin/python chapter-15-nlp/spam_data.py --download
.venv/bin/python chapter-15-nlp/encoder.py --download
```

- SMS Spam Collection v.1, T. A. Almeida and J. M. Gomez Hidalgo, UCI Machine
  Learning Repository, doi 10.24432/C5CC84. The authors provide it free of
  charge with no limitations other than their copyright and a warranty
  disclaimer, and ask users to cite their 2011 paper. The file is verified by
  its SHA-256 hash and is not stored in this repository.
- `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243`, Apache-2.0
  licence. Weights are cached in `data/` and are not stored in this repository.

## 4. Files supplied

- `based-on/spam_compare.py`: incomplete; data loading, the TF-IDF score,
  the training loop structure and `run()` are supplied.
- `based-on/test_spam_compare.py`: complete acceptance tests.
- `full-code/`: finished solution, tests and result verifier.
- `RESULTS.md`: values to record and questions to explain.

## 5. Steps

Complete the numbered places in `based-on/spam_compare.py`:

1. `labelled_subset`: draw a stratified sample of 200 messages from the
   training split with `train_test_split(..., train_size=size,
   stratify=labels, random_state=seed)`.
2. `frozen_score`: load the encoder, embed the 200 messages with `embed`, fit
   `LogisticRegression(class_weight='balanced', max_iter=2000)` on the
   embeddings, embed and predict the test messages, return macro-F1.
3. `logits`: tokenise with padding and truncation to 64 tokens, run the
   encoder, mean-pool `last_hidden_state` with the attention mask, apply `head`.
4. In the training loop: cross-entropy between the batch logits and
   `targets[idx]`, then `zero_grad`, `backward`, `step`.
5. Predict the test set in batches of 64 inside `torch.inference_mode()`, take
   the arg-max class, return macro-F1.
6. Run the tests, then the script, and fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-15-nlp/applied/based-on
.venv/bin/python chapter-15-nlp/applied/based-on/spam_compare.py
```

## 6. What to record

The counts, the three macro-F1 scores, and the mean and population standard
deviation of the fine-tuned score over seeds 42, 43 and 44. Also note the run
time of each method on your machine (not checked by the verifier). Three seeds
show the spread from training randomness only; they are not an uncertainty
estimate over data.

## 7. Verification

```bash
.venv/bin/python chapter-15-nlp/applied/full-code/verify_results.py chapter-15-nlp/applied/RESULTS.md
```

Every field must report `OK`. Counts must match exactly; scores may differ by
up to 0.01 across platforms. Written explanations require human assessment.

## 8. Troubleshooting

- `FileNotFoundError` naming `--download`: run the two download commands.
- `NotImplementedError`: complete the indicated step.
- Frozen score far below the reference: check that the embeddings are
  normalised mean-pooled vectors, not the first token's vector.
- Fine-tuned score near 0.47: the model predicts one class only; check that the
  optimiser step runs and that `model.train()` is called before training.
