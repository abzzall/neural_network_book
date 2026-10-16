# Chapter 1 — Introduction

Executable material for chapter 1 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `neuron_demo.py` | Listing 1.1 | One neuron decides whether to irrigate a field on three days |
| `test_neuron_demo.py` | — | Runs the demo and checks every printed line |
| `applied/` | Applied analysis | How the bias moves a neuron's decisions |

```bash
.venv/bin/python chapter-01-introduction/neuron_demo.py
.venv/bin/python -m pytest chapter-01-introduction
```

Expected output:

```text
dry and hot          z = +1.65, sigmoid = 0.839 -> irrigate
wet                  z = -1.90, sigmoid = 0.130 -> do not irrigate
dry, rain expected   z = -0.95, sigmoid = 0.279 -> do not irrigate
```

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
