# Clean-checkout validation

Before publishing, verify that a fresh copy of the repository works with only
the README's Quick start commands.

```bash
git clone https://github.com/abzzall/neural_network_book.git /tmp/nn-check
cd /tmp/nn-check
uv venv .venv
uv pip install -p .venv -r requirements.txt
uv pip install -p .venv -e .
./scripts/check-all.sh
```

Expected last line: `All companion-code checks passed.`

## Record

| Date | Python | uv | Result |
|---|---|---|---|
| 2026-10-16 | 3.14.4 | 0.12.10 | 4 passed; all checks passed (copy of the working tree, before publication) |
