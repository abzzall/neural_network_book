# Neural Networks — Samples and Code

Public repository: <https://github.com/abzzall/neural_network_book>

This repository contains the executable samples, applied exercises, and tests
for the *Neural Networks* textbook. The book explains the concepts and the
mathematics; this repository provides the runnable programs. Every listing
printed in the book is read directly from a file here, and every example is
run and tested before it is published.

## Repository rules

- Public commit messages, file contents, and documentation must not refer to
  collaboration tools, assistants, or code-generation systems.
- Commits must use `abzzall` as the author and committer identity.
- An example is added only after it has been run and its results checked by a
  test. Numbers quoted in the book come from running the code.
- Dependencies are pinned to exact versions. A library is added together with
  the first tested example that uses it.
- Keep virtual environments, caches, downloaded datasets, and model weights
  out of version control.

## Quick start

Verified with Python 3.14.4 and [uv](https://docs.astral.sh/uv/) 0.12.10 on
Linux, 2026-10-16:

```bash
git clone https://github.com/abzzall/neural_network_book.git
cd neural_network_book
uv venv .venv
uv pip install -p .venv -r requirements.txt
uv pip install -p .venv -e .       # makes `import nnbook` work everywhere
./scripts/check-all.sh             # runs every test
```

The last command ends with `All companion-code checks passed.` Run any
chapter's script from the repository root, for example:

```bash
.venv/bin/python chapter-04-perceptron/perceptron_demo.py
```

The book's appendix on setting up the environment explains each step.

## Structure

| Path | Contents |
|------|----------|
| `nnbook/` | A small neural-network library built **from scratch** in NumPy, growing chapter by chapter |
| `tests/` | Tests for `nnbook` |
| `chapter-NN-<topic>/` | Scripts for chapter NN, with their tests beside them; `applied/` holds the chapter's applied exercise |
| `scripts/check-all.sh` | The repository-wide check |
| `validation/` | How a clean checkout is verified |
| `data/` | Downloaded datasets (not committed) |

## Chapters

| # | Chapter | Folder |
|---|---------|--------|
| 1 | Introduction | [`chapter-01-introduction`](chapter-01-introduction) |
| 2 | Mathematical Foundations | [`chapter-02-math-foundations`](chapter-02-math-foundations) |
| 3 | Data: Requirements, Cleaning, and Preprocessing | [`chapter-03-data`](chapter-03-data) |
| 4 | The Perceptron | [`chapter-04-perceptron`](chapter-04-perceptron) |
| 5 | Multilayer Networks and Backpropagation | [`chapter-05-mlp-backprop`](chapter-05-mlp-backprop) |
| 6 | Optimising the Training | [`chapter-06-optimization`](chapter-06-optimization) |
| 7 | Regularisation and Model Evaluation | [`chapter-07-regularization`](chapter-07-regularization) |
| 8 | From NumPy to PyTorch | [`chapter-08-pytorch`](chapter-08-pytorch) |
| 9 | Convolutional Neural Networks | [`chapter-09-cnn`](chapter-09-cnn) |
| 10 | Recurrent Neural Networks | [`chapter-10-rnn`](chapter-10-rnn) |
| 11 | Attention and Transformers | [`chapter-11-transformers`](chapter-11-transformers) |
| 12 | Generative Models | [`chapter-12-generative`](chapter-12-generative) |
| 13 | Choosing a Model in Practice | [`chapter-13-model-selection`](chapter-13-model-selection) |

Chapters 4–7 build networks from scratch with NumPy. From chapter 8 the book
uses PyTorch.

## License

Code: [MIT](LICENSE.md). The book manuscript is not included in that grant.
