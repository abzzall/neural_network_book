#!/usr/bin/env bash
# Repository-wide check: every test, including the tests that run each
# chapter's scripts exactly as a reader would.
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_dir}"

python_bin="${PYTHON:-${repository_dir}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "No Python environment at ${python_bin}. See README.md, Quick start." >&2
  exit 1
fi

echo "==> Python: $("${python_bin}" --version)"
echo "==> Running all tests"
"${python_bin}" -m pytest -v --ignore-glob="*/based-on/*"   # based-on/ fails until a reader completes it

# Applied exercises: each finished solution marks itself.
shopt -s nullglob
for verifier in chapter-*/applied/full-code/verify_results.py; do
  echo "==> Verifying ${verifier}"
  "${python_bin}" "${verifier}"
done

echo "All companion-code checks passed."
