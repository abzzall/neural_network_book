"""Sentence embeddings from a pinned pretrained encoder (mean pooling)."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'          # Apache-2.0 licence
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CACHE = Path(__file__).resolve().parents[1] / 'data' / 'minilm-l6-v2-1110a24'
FILES = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.txt', 'special_tokens_map.json', '1_Pooling/config.json', 'README.md']


def load_encoder(download=False, directory=CACHE):
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=directory,
                          allow_patterns=FILES)
    if not (directory / 'model.safetensors').is_file():
        raise FileNotFoundError('Model missing: run encoder.py --download first')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModel.from_pretrained(directory, local_files_only=True,
                                      use_safetensors=True)
    return tokenizer, model.eval()


def mean_pool(hidden, mask):
    """Average token vectors, ignoring padding, eq. (15.8)."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.inference_mode()
def embed(tokenizer, model, texts, batch_size=64, max_length=128):
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True,
                          truncation=True, max_length=max_length, return_tensors='pt')
        hidden = model(**batch).last_hidden_state
        chunks.append(mean_pool(hidden, batch['attention_mask']))
    return torch.nn.functional.normalize(torch.cat(chunks), dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    tokenizer, model = load_encoder(parser.parse_args().download)
    vectors = embed(tokenizer, model, ['The train is late.', 'My train was delayed.'])
    print('embedding size:', vectors.shape[1])
    print(f'cosine similarity: {(vectors[0] @ vectors[1]).item():.4f}')
