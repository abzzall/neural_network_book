"""Optional SSDLite interface check on an original synthetic RGB image."""

import argparse
import hashlib
from pathlib import Path
import torch

URL = 'https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
SHA256 = 'a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2'
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / 'data' / 'vision-ssdlite'


def run(cache=DEFAULT_CACHE, download=False):
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
    checkpoint = Path(cache) / URL.rsplit('/', 1)[1]
    if not checkpoint.exists():
        if not download:
            raise FileNotFoundError('Missing checkpoint; rerun with --download')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL, str(checkpoint), hash_prefix=SHA256)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SHA256:
        raise ValueError('checkpoint SHA-256 mismatch')
    torch.set_num_threads(1)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None,
                                         num_classes=len(weights.meta['categories']))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    image = torch.full((3, 320, 320), .4)
    image[:, 80:200, 100:240] = .8
    with torch.inference_mode():
        prediction = model([weights.transforms()(image)])[0]
    count = len(prediction['scores'])
    assert prediction['boxes'].shape == (count, 4)
    assert prediction['labels'].shape == (count,)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    assert ((prediction['scores'] >= 0) & (prediction['scores'] <= 1)).all()
    return count, int((prediction['scores'] >= .5).sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    count, confident = run(args.cache, args.download)
    print('output schema valid: True')
    print(f'detections returned: {count}; scores >= 0.5: {confident}')
    print('Synthetic image: interface check only, no accuracy claim.')
