"""Compare an optional results JSON with a fresh local pretrained-model probe."""

from functools import lru_cache
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_probe
from pretrained_probe import CACHE


@lru_cache(maxsize=1)
def reference_values():
    return sentiment_probe.run()


def check(path):
    recorded = json.loads(Path(path).read_text())
    return recorded == reference_values()


if __name__ == '__main__':
    if not (CACHE / 'model.safetensors').is_file():
        print('SKIP: pretrained model absent; run chapter-14-transformers/pretrained_probe.py --download')
        sys.exit(0)
    print(json.dumps(reference_values(), indent=2))
    if len(sys.argv) > 1:
        ok = check(sys.argv[1])
        print('PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
