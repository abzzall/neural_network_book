"""A pinned CLIP model (LAION ViT-B/32, MIT licence); downloads only with --download."""

import argparse
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
REVISION = '1a25a446712ba5ee05982a381eed697ef9b435cf'
CACHE = Path(__file__).resolve().parents[1] / 'data' / 'clip-vit-b-32-laion-1a25a44'
FILES = ['config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
         'README.md']


def load_clip(download=False, directory=CACHE):
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=directory, allow_patterns=FILES)
    if not (directory / 'model.safetensors').is_file():
        raise FileNotFoundError('Model missing: run clip_model.py --download first')
    processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)
    model = CLIPModel.from_pretrained(directory, local_files_only=True, use_safetensors=True)
    return processor, model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    processor, model = load_clip(parser.parse_args().download)
    print('parameters:', sum(p.numel() for p in model.parameters()))
