"""Topic classification of Kazakh sentences, and of the same sentences in English.

Data: SIB-200 (Adelani et al., 2024), CC BY-SA 4.0: sentences of the Flores-200
benchmark labelled with seven topics, the same sentences in every language.
Encoder: multilingual-e5-small (MIT). Download both once (about 470 MB):
    python kazakh_topics.py --download
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parents[1] / 'data'
SIB = ('Davlan/sib200', '38977a667f6fc264d5c26ec57a01e16db040b358', DATA / 'sib200-38977a6')
E5 = ('intfloat/multilingual-e5-small', '614241f622f53c4eeff9890bdc4f31cfecc418b3',
      DATA / 'multilingual-e5-small-614241f')
LANGUAGES = {'Kazakh': 'kaz_Cyrl', 'English': 'eng_Latn'}


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(SIB[0], repo_type='dataset', revision=SIB[1], local_dir=SIB[2],
                      allow_patterns=[f'data/{c}/*' for c in LANGUAGES.values()])
    snapshot_download(E5[0], revision=E5[1], local_dir=E5[2],
                      allow_patterns=['*.json', 'model.safetensors', 'sentencepiece.bpe.model'])


def load(code, split):
    table = pd.read_csv(SIB[2] / 'data' / code / f'{split}.tsv', sep='\t')
    return table['text'].tolist(), table['category'].to_numpy()


def embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(E5[2], local_files_only=True)
    model = AutoModel.from_pretrained(E5[2], local_files_only=True).eval()
    batch = tokenizer(['query: ' + t for t in texts], padding=True, truncation=True,
                      max_length=128, return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
    mask = batch['attention_mask'][..., None]
    vectors = (hidden * mask).sum(1) / mask.sum(1)            # mean pooling, eq. (15.5)
    return torch.nn.functional.normalize(vectors, dim=1).numpy()


def score(features, y_train, y_test):
    model = LogisticRegression(max_iter=5000, C=10.0).fit(features[0], y_train)
    return f1_score(y_test, model.predict(features[1]), average='macro')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    if parser.parse_args().download:
        download()
    for language, code in LANGUAGES.items():
        (train, y_train), (test, y_test) = load(code, 'train'), load(code, 'test')
        words = TfidfVectorizer().fit(train)
        chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5)).fit(train)
        results = {'TF-IDF words': score([words.transform(train), words.transform(test)],
                                         y_train, y_test),
                   'TF-IDF characters': score([chars.transform(train),
                                               chars.transform(test)], y_train, y_test),
                   'multilingual encoder': score([embed(train), embed(test)], y_train, y_test)}
        analyse = words.build_analyzer()
        test_words = [w for t in test for w in analyse(t)]
        unseen = np.mean([w not in words.vocabulary_ for w in test_words])
        print(f'{language}: {len(train)} training and {len(test)} test sentences, '
              f'vocabulary {len(words.vocabulary_)} words, '
              f'{unseen:.1%} of test words unseen in training')
        for name, value in results.items():
            print(f'  {name:21s} macro-F1 {value:.4f}')
