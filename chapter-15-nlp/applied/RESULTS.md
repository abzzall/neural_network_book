# Chapter 15 applied exercise — results

Fill the Value column with four or more significant figures.

| Quantity | Value |
|---|---|
| `labelled_messages` | |
| `labelled_spam` | |
| `test_messages` | |
| `tfidf_macro_f1` | |
| `frozen_macro_f1` | |
| `finetuned_macro_f1_mean` | |
| `finetuned_macro_f1_std` | |

Run times on your machine (not checked): TF-IDF ____ s, frozen ____ s,
fine-tuning ____ s per seed.

## What to explain

1. With only 25 spam examples among the 200 labels, why can frozen embeddings
   beat TF-IDF, which learns its vocabulary from the same 200 messages?
2. Why does fine-tuning beat the frozen encoder? What exactly changes in the
   model during fine-tuning?
3. With all 3878 training labels, TF-IDF reaches macro-F1 0.9514
   (`tfidf_baseline.py`). Which method would you deploy for an SMS gateway that
   processes millions of messages a day, and why?
4. Why is macro-F1 reported rather than accuracy for this dataset?
