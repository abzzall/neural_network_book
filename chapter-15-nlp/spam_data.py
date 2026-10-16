"""Download, load, and split the SMS Spam Collection (UCI, doi 10.24432/C5CC84)."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile

from sklearn.model_selection import train_test_split

URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
SHA256 = '7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d'
PATH = Path(__file__).resolve().parents[1] / 'data' / 'sms-spam' / 'SMSSpamCollection'


def download(path=PATH):
    with urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    raw = archive.read('SMSSpamCollection')
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Downloaded file differs from the verified version')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def load(path=PATH):
    if not path.is_file():
        raise FileNotFoundError('Data missing: run spam_data.py --download first')
    rows = [line.split('\t', 1) for line in path.read_text('utf-8').splitlines()]
    return [text for _, text in rows], [int(label == 'spam') for label, _ in rows]


def split(texts, labels, deduplicate=True, seed=42, test_size=0.25):
    if deduplicate:                              # keep the first copy of each message
        first = {}
        for text, label in zip(texts, labels):
            first.setdefault(text, label)
        texts, labels = list(first), list(first.values())
    return train_test_split(texts, labels, test_size=test_size,
                            stratify=labels, random_state=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    if parser.parse_args().download:
        download()
    texts, labels = load()
    print(f'messages: {len(texts)}, spam: {sum(labels)}')
    print(f'distinct messages: {len(set(texts))}')
    for name, dedup in (('with duplicates', False), ('deduplicated', True)):
        train, test, _, _ = split(texts, labels, deduplicate=dedup)
        seen = sum(text in set(train) for text in test)
        print(f'{name}: train {len(train)}, test {len(test)}, test also in train {seen}')
