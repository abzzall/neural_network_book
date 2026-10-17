"""Download and load the German Credit data (UCI Statlog, doi 10.24432/C5NC77, CC BY 4.0)."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile

import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip'
SHA256 = 'b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871'
PATH = Path(__file__).resolve().parents[1] / 'data' / 'german-credit' / 'german.data'
COLUMNS = ['checking', 'duration', 'history', 'purpose', 'amount', 'savings',
           'employment', 'instalment_rate', 'status_sex', 'guarantors', 'residence',
           'property', 'age', 'other_plans', 'housing', 'existing_credits', 'job',
           'dependants', 'telephone', 'foreign_worker', 'target']


def download(path=PATH):
    with urlopen(URL, timeout=60) as response:
        raw = zipfile.ZipFile(io.BytesIO(response.read())).read('german.data')
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Downloaded file differs from the verified version')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def load(path=PATH):
    """Features, label good = 1 / bad = 0, and the group young (age <= 25)."""
    if not path.is_file():
        raise FileNotFoundError('Data missing: run credit_data.py --download first')
    table = pd.read_csv(path, sep=' ', header=None, names=COLUMNS)
    good = (table.pop('target') == 1).astype(int)
    young = (table['age'] <= 25).astype(int)
    return table, good, young


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    if parser.parse_args().download:
        download()
    X, good, young = load()
    print(f'applicants: {len(X)}, features: {X.shape[1]}, good credit: {good.sum()}')
    for name, mask in (('aged 25 or less', young == 1), ('older than 25', young == 0)):
        print(f'{name}: {mask.sum()} applicants, share with good credit {good[mask].mean():.3f}')
