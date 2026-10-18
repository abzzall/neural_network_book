# Neural Networks — Samples and Code

Public repository: <https://github.com/abzzall/neural_network_book>

This repository contains the executable samples, applied exercises, and tests
for the *Neural Networks* textbook. The book explains the concepts and the
mathematics; this repository provides the runnable programs. Every listing
printed in the book is read directly from a file here, and every example is
run and tested before it is published.

## How the code is checked

- An example is added only after it has been run and its results checked by a
  test. Numbers quoted in the book come from running the code.
- Dependencies are pinned to exact versions. A library is added together with
  the first tested example that uses it.
- Virtual environments, caches, downloaded datasets, and model weights are
  kept out of version control.

## Quick start

Verified with Python 3.14.4 and [uv](https://docs.astral.sh/uv/) 0.12.10 on
Linux, 2026-10-17, in a fresh clone:

```bash
git clone https://github.com/abzzall/neural_network_book.git
cd neural_network_book
uv venv .venv
uv pip install -p .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -r requirements.txt
uv pip install -p .venv -e .       # makes `import nnbook` work everywhere
./scripts/check-all.sh             # runs every test
```

PyTorch's CPU build comes from its own index, everything else from PyPI; the
second install keeps `torch 2.14.1+cpu`. Do not merge the two installs into one
command with `--extra-index-url`: the PyTorch index also hosts `numpy`, but not
2.5.4, and uv then reports the requirements as unsatisfiable. Appendix B of the
book explains the setup step by step, including plain `pip`.

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
| 15 | Working with Text | [`chapter-15-nlp`](chapter-15-nlp) |
| 16 | Time-Series Data | [`chapter-16-time-series`](chapter-16-time-series) |
| 17 | Computer Vision beyond Classification | [`chapter-17-vision`](chapter-17-vision) |
| 18 | Graph Neural Networks | [`chapter-18-gnn`](chapter-18-gnn) |
| 19 | Deep Reinforcement Learning | [`chapter-19-rl`](chapter-19-rl) |
| 20 | Generative Models | [`chapter-20-generative`](chapter-20-generative) |
| 21 | Large Language Models | [`chapter-21-llm`](chapter-21-llm) |
| 22 | Pretrained Models and Transfer Learning | [`chapter-22-pretrained`](chapter-22-pretrained) |
| 23 | Interpretability and Fairness | [`chapter-23-interpretability`](chapter-23-interpretability) |
| 24 | Deployment and MLOps | [`chapter-24-deployment`](chapter-24-deployment) |
| 25 | Choosing a Model in Practice | [`chapter-25-model-selection`](chapter-25-model-selection) |
| 26 | Capstone Project | [`chapter-26-capstone`](chapter-26-capstone) |

Chapters 5–8 build networks from scratch with NumPy. From chapter 9 the book
uses PyTorch.

Chapter 14 adds [attention and transformer examples](chapter-14-transformers/README.md),
including an optional, explicitly downloaded pretrained sentiment checkpoint.
Routine tests never download weights and report skips when that checkpoint is absent.

Chapter 15 adds [text-processing examples](chapter-15-nlp/README.md). Its
SMS Spam Collection data, pinned sentence encoder and the multilingual XLM-R
tokeniser are downloaded only on request (`--download`); tests that need them
report skips until then.

Downloads and optional packages of later chapters, all requested explicitly
and none stored in this repository:

| Chapter | Download (`--download`) | Optional requirements file |
|---|---|---|
| 16 | UCI Bike Sharing data; Chronos-Bolt tiny | `requirements-chronos.txt` |
| 17 | TorchVision SSDLite weights | `chapter-17-vision/requirements.txt` |
| 19 | none | `gymnasium==1.4.0` (comparison with Gymnasium's cart-pole) |
| 21 | GPT-2, Qwen2.5-0.5B-Instruct | none |
| 22 | LAION CLIP ViT-B/32 (and the chapter 15 data and encoder) | none |
| 23, 25 | UCI German Credit data | none |
| 24 | none | `requirements-deployment.txt` |

Each chapter README states the licence or terms of its data and weights. Most
are MIT, Apache-2.0, BSD or CC BY 4.0; the SMS Spam Collection is free to use
with a citation request, and the TorchVision weights of chapter 17 come with
the separate terms of their training data, as that README explains.

Chapter 11 adds optional CUDA examples. Its [environment instructions](chapter-11-cuda/README.md)
describe the verified GPU build, compiler prerequisites and CPU-only skip behaviour.

## License

Code: [MIT](LICENSE.md). The book manuscript is not included in that grant.
