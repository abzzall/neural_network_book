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
uv pip install -p .venv -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -e .       # makes `import nnbook` work everywhere
./scripts/check-all.sh             # runs every test
```

The last command ends with `All companion-code checks passed.` Run any
chapter's script from the repository root, for example:

```bash
.venv/bin/python chapter-05-linear-models/perceptron_demo.py
```

The extra index installs the CPU-only build of PyTorch, which is all the examples
need and much smaller than the default build; without it the default build of the
same version is installed. The book's appendix on setting up the environment
explains each step.

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
| 2 | Linear Algebra and Calculus | [`chapter-02-linear-algebra`](chapter-02-linear-algebra) |
| 3 | Probability, Statistics, and Information Theory | [`chapter-03-probability`](chapter-03-probability) |
| 4 | Data: Requirements, Cleaning, and Preprocessing | [`chapter-04-data`](chapter-04-data) |
| 5 | Linear Models and the Perceptron | [`chapter-05-linear-models`](chapter-05-linear-models) |
| 6 | Multilayer Networks and Backpropagation | [`chapter-06-mlp-backprop`](chapter-06-mlp-backprop) |
| 7 | Optimising the Training | [`chapter-07-optimization`](chapter-07-optimization) |
| 8 | Regularisation and Model Evaluation | [`chapter-08-regularization`](chapter-08-regularization) |
| 9 | From NumPy to PyTorch | [`chapter-09-pytorch`](chapter-09-pytorch) |
| 10 | Automatic Differentiation from Scratch | [`chapter-10-autograd`](chapter-10-autograd) |
| 11 | GPU Computing with CUDA | [`chapter-11-cuda`](chapter-11-cuda) |
| 12 | Convolutional Neural Networks | [`chapter-12-cnn`](chapter-12-cnn) |
| 13 | Recurrent Neural Networks | [`chapter-13-rnn`](chapter-13-rnn) |
| 14 | Attention and Transformers | [`chapter-14-transformers`](chapter-14-transformers) |
| 15 | Generative Models | [`chapter-15-generative`](chapter-15-generative) |
| 16 | Choosing a Model in Practice | [`chapter-16-model-selection`](chapter-16-model-selection) |
| 17 | Capstone Project | [`chapter-17-capstone`](chapter-17-capstone) |

Chapters 5–8 build networks from scratch with NumPy. From chapter 9 the book
uses PyTorch.

Chapter 11 adds optional CUDA examples. Its [environment instructions](chapter-11-cuda/README.md)
describe the verified GPU build, compiler prerequisites and CPU-only skip behaviour.

## License

Code: [MIT](LICENSE.md). The book manuscript is not included in that grant.
