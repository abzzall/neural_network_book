"""The Kazakh sentence of tokenizer_coverage.py, read by a multilingual tokeniser.

Only the tokeniser of XLM-R base (MIT licence) is downloaded, about 14 MB,
at a pinned revision: python multilingual_coverage.py --download
"""

import argparse
from pathlib import Path

MODEL = 'FacebookAI/xlm-roberta-base'
REVISION = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
CACHE = Path(__file__).resolve().parents[1] / 'data' / 'xlm-roberta-base-tokenizer-e73636d'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'sentencepiece.bpe.model']

SENTENCES = {
    'English': 'Kazakhstan is a country in Central Asia.',
    # The same statement in Kazakh, written with escapes to keep this file ASCII.
    'Kazakh': ('\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d '
               '\u041e\u0440\u0442\u0430\u043b\u044b\u049b '
               '\u0410\u0437\u0438\u044f\u0434\u0430\u0493\u044b '
               '\u043c\u0435\u043c\u043b\u0435\u043a\u0435\u0442.'),
}


def load_tokenizer(download=False):
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=CACHE, allow_patterns=FILES)
    if not (CACHE / 'tokenizer.json').is_file():
        raise FileNotFoundError('Tokeniser missing: run multilingual_coverage.py --download')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CACHE, local_files_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    tokenizer = load_tokenizer(parser.parse_args().download)
    print(f'vocabulary size: {tokenizer.vocab_size}')
    for language, sentence in SENTENCES.items():
        tokens = tokenizer.tokenize(sentence)
        unknown = tokens.count(tokenizer.unk_token)
        print(f'{language}: {len(sentence.split())} words -> {len(tokens)} tokens, '
              f'{unknown} unknown')
        print('  ' + ' '.join(tokens))
