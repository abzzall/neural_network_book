# Authoring standard

This standard applies to every example and applied exercise in this repository.

## Every example is run, tested, and verified

An example shows readers how code works. A listing that does not run, or a
number the code does not produce, misleads the reader. Therefore:

- every script has a test that runs it the way a reader would and checks its
  output (see `chapter-05-linear-models/test_perceptron_demo.py`);
- every number printed in the book, including in worked solutions, is produced
  by running the code, never computed by hand or estimated;
- results are deterministic: every random generator is seeded, and examples run
  on a CPU in reasonable time, so every reader gets the printed numbers;
- `./scripts/check-all.sh` must pass before anything is committed.

Never describe an example as tested unless it was executed.

## Exercises do not use screenshots

Evidence is a recorded value (a loss, an accuracy, a parameter, a shape) or a
written explanation. A recorded value is checkable by a script; a picture is
not.

## Required exercise files

Each chapter's applied exercise lives in `chapter-NN-<topic>/applied/`:

- `INSTRUCTIONS.md` — the student-facing procedure;
- `RESULTS.md` — a template listing every value to record and every question
  to answer in prose;
- `based-on/` — the starting point: tests complete, code stubbed behind
  numbered steps that match the chapter's steps; it fails until completed;
- `full-code/` — the finished result, which passes its own tests;
- `full-code/verify_results.py` — checks the recorded values mechanically.

Only `full-code/` is checked by `scripts/check-all.sh`; `based-on/` is expected
to fail until the reader completes it.

## Required structure of `INSTRUCTIONS.md`

1. **Outcome** — what the student will have working at the end.
2. **Verified versions** — the exact environment used, with a date.
3. **Files supplied.**
4. **Steps** — numbered, single-action steps.
5. **What to record** — the values that go into `RESULTS.md`.
6. **What to explain** — the questions answered in prose.
7. **Verification** — how the student, or a marker, checks the recorded values.
8. **Troubleshooting.**

## How much to explain: three tiers

**Tier 1 — neural networks and their mathematics. Explain fully.**
Models, losses, gradients, backpropagation, optimisers, regularisation,
evaluation, data preparation. These are the subject. Explain what each does,
why, and what it does not guarantee. Never reduce one to a command to copy.

**Tier 2 — not the subject, but where readers get stuck. Explain.**
Tensor and array shapes, broadcasting, dtypes and devices, random seeds and
reproducibility, data loading and batching, train/evaluation modes, numerical
stability. The test: getting it wrong produces a *wrong* result, often silently.

**Tier 3 — incidental. Supply whole, do not explain.**
Plot styling, file and directory boilerplate, an install that is merely an
install. If getting it wrong produces an ugly result rather than a wrong one,
it is tier 3.

## Steps are atomic

A step is one action with one justification, never "now write the training
loop". Each line of a finished file is justified once, in the order a person
would write it. Tier 3 material is handed over whole.

## Installation is explained once

The first chapter that needs a library explains how to install it and how to
tell that it worked. Later chapters refer back.

## Versions

Pin exact versions in `requirements.txt` and `pyproject.toml`. Never write a
floating version. Record the verified Python version and date. Use only
current, non-deprecated APIs; consult the library's official documentation.

## Data and models

Use small public datasets whose licences allow educational use, downloaded by
a script into `data/` (not committed). State each dataset's source and licence.
Pretrained models are referenced by name and version, never committed.

## Publication readiness

A clean checkout must be able to follow the README's Quick start and pass
`./scripts/check-all.sh`. See `validation/clean-checkout.md`.
