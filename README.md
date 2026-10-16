# Нейронные сети — код к учебнику

Companion repository of the textbook **«Нейронные сети»**. It contains the complete,
runnable source code of every example and exercise in the book.

Официальный репозиторий учебника «Нейронные сети»: полный исходный код всех примеров книги.

## Quick start

```bash
git clone https://github.com/USERNAME/neural-networks-book.git
cd neural-networks-book
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import nnbook` work everywhere

pytest                             # run the tests
python chapters/ch03_perceptron/perceptron_demo.py
```

## Structure

| Path | Contents |
|------|----------|
| `nnbook/` | Small neural-network library built **from scratch** in NumPy, chapter by chapter |
| `chapters/chNN_<topic>/` | Scripts and notebooks for chapter NN of the book |
| `tests/` | `pytest` tests for `nnbook` |
| `data/` | Downloaded datasets (not committed) |

## Chapters

| # | Chapter | Code |
|---|---------|------|
| 1 | Введение | — |
| 2 | Математические основы | — |
| 3 | Перцептрон | [`chapters/ch03_perceptron`](chapters/ch03_perceptron) |
| 4 | Многослойные сети и обратное распространение | in progress |
| 5 | Оптимизация обучения | in progress |
| 6 | Регуляризация и обобщение | in progress |
| 7 | От NumPy к PyTorch | in progress |
| 8 | Свёрточные нейронные сети | in progress |
| 9 | Рекуррентные нейронные сети | in progress |
| 10 | Механизм внимания и трансформеры | in progress |
| 11 | Генеративные модели | in progress |

## Requirements

Python ≥ 3.11. Chapters 3–6 need only NumPy and Matplotlib; PyTorch is required from chapter 7.

## License

Code: [MIT](LICENSE).
