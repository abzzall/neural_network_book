"""Pinned pretrained language models; downloads only with --download."""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(__file__).resolve().parents[1] / 'data'
MODELS = {   # name: (Hub id, revision, licence)
    'gpt2': ('openai-community/gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'MIT'),
    'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775',
             'Apache-2.0'),
}
FILES = ['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json', 'merges.txt', 'LICENSE', 'README.md']


def directory(name):
    hub_id, revision, _ = MODELS[name]
    return DATA / f'{hub_id.split("/")[1].lower()}-{revision[:7]}'


def download(name):
    from huggingface_hub import snapshot_download
    hub_id, revision, _ = MODELS[name]
    snapshot_download(hub_id, revision=revision, local_dir=directory(name),
                      allow_patterns=FILES)


def load(name, folder=None):
    folder = folder or directory(name)
    if not (folder / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Model missing: run lm.py --download {name} first')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
                                                 use_safetensors=True)
    return tokenizer, model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', nargs='+', choices=sorted(MODELS), default=[])
    for name in parser.parse_args().download:
        download(name)
    for name in MODELS:
        ok = (directory(name) / 'model.safetensors').is_file()
        print(f'{name}: {"present" if ok else "missing"} ({MODELS[name][2]} licence)')
