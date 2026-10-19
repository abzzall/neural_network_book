"""Chapter 22: frozen ImageNet features transferred to EuroSAT.

The experiment compares the same logistic-regression head on features from a
pretrained ResNet-18 and an untrained ResNet-18. This isolates what pretraining
contributes; it is not a state-of-the-art EuroSAT benchmark.

Data: EuroSAT RGB (Helber et al. 2019, MIT licence, Zenodo record 7711810),
27,000 Sentinel-2 image patches in 10 land-use classes. Model: TorchVision's
ResNet-18 ImageNet-1K V1 weights (BSD-3-Clause).
"""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).resolve().parents[1] / 'data'
EUROSAT = DATA / 'eurosat-rgb-7711810'
ZIP_URL = 'https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content'
ZIP_MD5 = 'f46e308c4d50d4bf32fedad2d3d62f3b'


def safe_extract(archive, destination):
    """Extract an archive only when every member remains below destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'unsafe archive member: {member.filename}')
        zf.extractall(destination)


def download():
    """Download and verify EuroSAT, then cache TorchVision's pinned weights."""
    EUROSAT.mkdir(parents=True, exist_ok=True)
    archive = EUROSAT / 'EuroSAT_RGB.zip'
    if not archive.exists():
        urllib.request.urlretrieve(ZIP_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != ZIP_MD5:
        raise RuntimeError(f'{archive} does not match the published checksum')
    root = EUROSAT / '2750'
    if not root.is_dir():
        safe_extract(archive, EUROSAT)

    from torchvision.models import ResNet18_Weights, resnet18
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return image_root()


def image_root():
    """Return the extracted class-folder root or explain how to obtain it."""
    candidates = (EUROSAT / '2750', EUROSAT / 'EuroSAT_RGB', EUROSAT)
    for root in candidates:
        if not root.is_dir():
            continue
        classes = [p for p in root.iterdir()
                   if p.is_dir() and any(p.glob('*.jpg'))]
        if classes:
            return root
    raise FileNotFoundError(
        f'EuroSAT images not found below {EUROSAT}; run this script with --download')


def stratified_files(root, train_per_class, test_per_class, seed=0):
    """Make a reproducible, disjoint, class-balanced split of image paths."""
    if train_per_class < 1 or test_per_class < 1:
        raise ValueError('train_per_class and test_per_class must be positive')
    rng = np.random.default_rng(seed)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    train, test = [], []
    for label, name in enumerate(classes):
        paths = sorted((root / name).glob('*.jpg'))
        required = train_per_class + test_per_class
        if len(paths) < required:
            raise ValueError(f'{name} has {len(paths)} images; {required} required')
        chosen = rng.permutation(len(paths))[:required]
        train.extend((paths[i], label) for i in chosen[:train_per_class])
        test.extend((paths[i], label) for i in chosen[train_per_class:])
    return classes, train, test


def extract_features(samples, model, transform, device, batch_size):
    """Transform images and return one feature row and label per sample."""
    import torch

    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            images = torch.stack([
                transform(Image.open(path).convert('RGB')) for path, _ in batch
            ]).to(device)
            features.append(model(images).cpu().numpy())
            labels.extend(label for _, label in batch)
    return np.concatenate(features), np.asarray(labels)


def representations(pretrained, device):
    """Construct a ResNet-18 feature extractor and its prescribed transform."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    torch.manual_seed(0)
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model.to(device), transform


def score_features(train, test, pretrained, device, batch_size):
    model, transform = representations(pretrained, device)
    x_train, y_train = extract_features(train, model, transform, device, batch_size)
    x_test, y_test = extract_features(test, model, transform, device, batch_size)
    classifier = LogisticRegression(max_iter=2000, C=1.0)
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='download data and weights')
    parser.add_argument('--train-per-class', type=int, default=20)
    parser.add_argument('--test-per-class', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat-seeds', type=int, nargs='+', help='report paired split repeats')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download:
        download()
    root = image_root()
    classes, train, test = stratified_files(
        root, args.train_per_class, args.test_per_class, args.seed)

    import torch
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    torch.set_num_threads(4)
    if args.repeat_seeds:
        for seed in args.repeat_seeds:
            _, train, test = stratified_files(root, args.train_per_class, args.test_per_class, seed)
            random_acc = score_features(train, test, False, device, args.batch_size)
            pretrained_acc = score_features(train, test, True, device, args.batch_size)
            print(f'split {seed}: random {random_acc:.4f}, pretrained {pretrained_acc:.4f}, '
                  f'gain {pretrained_acc - random_acc:+.4f}', flush=True)
        return
    random_accuracy = score_features(train, test, False, device, args.batch_size)
    pretrained_accuracy = score_features(train, test, True, device, args.batch_size)
    print(f'classes: {len(classes)}, train: {len(train)}, test: {len(test)}')
    print(f'random frozen features:     {random_accuracy:.4f}')
    print(f'pretrained frozen features: {pretrained_accuracy:.4f}')
    print(f'transfer gain:              {pretrained_accuracy - random_accuracy:+.4f}')


if __name__ == '__main__':
    main()
