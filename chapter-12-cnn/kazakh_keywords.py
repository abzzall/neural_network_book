"""Chapter 12: spoken Kazakh commands, as log-mel spectrograms, classified by a small CNN.

Data: Kazakh Speech Commands Dataset (ISSAI), MIT licence: 3623 one-second
recordings of 35 commands by 119 speakers. Download once (75 MB):
    python kazakh_keywords.py --download
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn

REPO, REVISION = 'issai/kazakh-speech-commands', '8efc3b804520cecdcc38e5bc935f0575c7c25517'
FOLDER = Path(__file__).resolve().parents[1] / 'data' / 'kazakh-speech-commands-8efc3b8'


def download():
    from huggingface_hub import hf_hub_download
    archive = hf_hub_download(REPO, 'data.zip', repo_type='dataset', revision=REVISION,
                              local_dir=FOLDER)
    zipfile.ZipFile(archive).extractall(FOLDER)


def mel_filters(n_mels=40, n_fft=400, rate=16000):
    """Triangular filters equally spaced on the mel scale, m = 2595 log10(1 + f/700)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(0), mel(rate / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / rate)
    rising = (bins[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - bins[None]) / (edges[2:, None] - edges[1:-1, None])
    return np.clip(np.minimum(rising, falling), 0, None)


def log_mel(wave):
    """25 ms windows every 10 ms -> power spectrum -> 40 mel bands -> logarithm."""
    spectrum = torch.stft(torch.tensor(wave), n_fft=400, hop_length=160,
                          window=torch.hann_window(400), return_complex=True)
    power = spectrum.abs().numpy() ** 2                    # (201 frequencies, 101 frames)
    return np.log(mel_filters() @ power + 1e-6).astype(np.float32)


def load():
    classes = sorted(p.name for p in (FOLDER / 'data').iterdir())
    waves, labels = [], []
    for k, name in enumerate(classes):
        for f in sorted((FOLDER / 'data' / name).glob('*.wav')):
            x = wavfile.read(f)[1].astype(np.float32) / 32768
            waves.append(np.pad(x, (0, 16000 - len(x))))
            labels.append(k)
    return np.array(waves), np.array(labels), classes


def cnn(n_classes=35):
    block = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
    return nn.Sequential(*block(1, 32), nn.MaxPool2d(2), *block(32, 64), nn.MaxPool2d(2),
                         *block(64, 128), nn.MaxPool2d(2), *block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(),     # global average pooling
                         nn.Dropout(0.3), nn.Linear(128, n_classes))


def add_noise(waves, snr_db, seed=0):
    """White noise at a given signal-to-noise ratio, per recording."""
    noise = np.random.default_rng(seed).normal(size=waves.shape).astype(np.float32)
    power = (waves ** 2).mean(axis=1, keepdims=True)
    return waves + noise * np.sqrt(power / 10 ** (snr_db / 10))


def features(waves):
    """Log-mel spectrograms, each standardised by its own mean and spread."""
    S = np.stack([log_mel(w) for w in waves])
    S = (S - S.mean(axis=(1, 2), keepdims=True)) / S.std(axis=(1, 2), keepdims=True)
    return torch.tensor(S)[:, None]


def train(X, y, epochs=30, seed=0):
    torch.manual_seed(seed)
    model, generator = cnn(), torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = epochs * ((len(y) + 63) // 64)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 2e-3, total_steps=steps)
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(y), generator=generator).split(64):
            shift = int(torch.randint(-10, 11, (1,), generator=generator))
            optimiser.zero_grad()                      # shift up to 0.1 s in time: augmentation
            loss = nn.functional.cross_entropy(model(torch.roll(X[batch], shift, dims=3)),
                                               y[batch])
            loss.backward()
            optimiser.step()
            schedule.step()
    return model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--figures', help='folder for the spectrogram figure data')
    args = parser.parse_args()
    if args.download:
        download()
    torch.set_num_threads(8)
    waves, labels, classes = load()
    train_ids, test_ids = train_test_split(np.arange(len(labels)), test_size=0.2,
                                           stratify=labels, random_state=0)
    X, y = features(waves), torch.tensor(labels)
    print(f'{len(train_ids)} training and {len(test_ids)} test recordings of '
          f'{len(classes)} commands, spectrogram {tuple(X.shape[2:])}')
    averaged = LogisticRegression(max_iter=5000).fit(X[train_ids].mean(3)[:, 0], y[train_ids])
    print(f'baseline on the time-averaged spectrum: accuracy '
          f'{averaged.score(X[test_ids].mean(3)[:, 0], y[test_ids]):.4f}')
    model = train(X[train_ids], y[train_ids])
    with torch.inference_mode():
        predicted = model(X[test_ids]).argmax(1)
        print(f'CNN on log-mel spectrograms: accuracy '
              f'{(predicted == y[test_ids]).float().mean():.4f}')
        for snr in (20, 10, 0):
            noisy = features(add_noise(waves[test_ids], snr))
            accuracy = (model(noisy).argmax(1) == y[test_ids]).float().mean()
            print(f'  with white noise at {snr:2d} dB: accuracy {accuracy:.4f}')
    truth = y[test_ids].tolist()
    wrong = [(classes[a], classes[b]) for a, b in zip(truth, predicted.tolist()) if a != b]
    pair, count = max(((p, wrong.count(p)) for p in set(wrong)), key=lambda t: (t[1], t[0]))
    print(f'most frequent confusion: "{pair[0]}" heard as "{pair[1]}" ({count} times)')
    if args.figures:                                   # one spectrogram of each word
        for word in pair:
            k = test_ids[labels[test_ids] == classes.index(word)][0]
            S = log_mel(waves[k])[:, ::2]                # every second frame: 20 ms steps
            rows = [(2 * t, m, S[m, t]) for m in range(S.shape[0]) for t in range(S.shape[1])]
            np.savetxt(Path(args.figures) / f'spectrogram_{word}.dat', np.array(rows),
                       fmt='%.4g', header='t m v', comments='')
