"""Probe a pinned sentiment model; downloads require the explicit flag."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
REVISION = '714eb0fa89d2f80546fda750413ed43d93601a13'
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'distilbert-sst2-714eb0f'


def load_classifier(download=False, directory=CACHE):
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=directory,
                          allow_patterns=['config.json', 'model.safetensors',
                                          'tokenizer_config.json', 'vocab.txt', 'README.md'])
    if not (directory / 'model.safetensors').is_file():
        raise FileNotFoundError('Model missing: run pretrained_probe.py --download first')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        directory, local_files_only=True, use_safetensors=True)
    model.eval()
    return pipeline('text-classification', model=model, tokenizer=tokenizer, device=-1)


def positive_scores(classifier, texts):
    predictions = classifier(texts, top_k=None, batch_size=8, truncation=True, max_length=128)
    return [next(item['score'] for item in row if item['label'] == 'POSITIVE')
            for row in predictions]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    classifier = load_classifier(args.download)
    texts = ['The acting was excellent and the story moved me.',
             'The plot was dull and the ending disappointed me.']
    for text, score in zip(texts, positive_scores(classifier, texts)):
        print(f'positive probability: {score:.6f} | {text}')
