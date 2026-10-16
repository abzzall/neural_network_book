"""Download and load daily bike rentals (UCI Bike Sharing, doi 10.24432/C5W894, CC BY 4.0)."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
SHA256 = 'a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be'
PATH = Path(__file__).resolve().parents[1] / 'data' / 'bike-sharing' / 'day.csv'


def download(path=PATH):
    with urlopen(URL, timeout=60) as response:
        raw = zipfile.ZipFile(io.BytesIO(response.read())).read('day.csv')
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Downloaded file differs from the verified version')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def load(path=PATH):
    """Daily table indexed by date: rentals `cnt` plus calendar and weather columns."""
    if not path.is_file():
        raise FileNotFoundError('Data missing: run bike_data.py --download first')
    table = pd.read_csv(path, parse_dates=['dteday']).set_index('dteday')
    return table[['cnt', 'workingday', 'holiday', 'temp']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    if parser.parse_args().download:
        download()
    table = load()
    y = table['cnt'].to_numpy(float)
    print(f'days: {len(y)}, from {table.index[0].date()} to {table.index[-1].date()}')
    print(f'mean rentals 2011: {y[:365].mean():.1f}, 2012: {y[365:].mean():.1f}')
    by_day = table.groupby(table.index.dayofweek)['cnt'].mean()
    print('mean by weekday (Mon..Sun):', ' '.join(f'{v:.0f}' for v in by_day))
    low = int(np.argmin(y))
    print(f'lowest day: {table.index[low].date()} with {y[low]:.0f} rentals')
