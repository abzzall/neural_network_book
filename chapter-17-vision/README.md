# Chapter 17 — Computer Vision beyond Classification

Run commands from the companion repository root with its existing environment.
All standard examples run on the CPU. The optional detector requires the pinned
TorchVision dependency; the rest use dependencies introduced earlier.

| File | Purpose |
|---|---|
| `iou_nms.py` | Continuous-coordinate IoU and stable, single-class suppression |
| `average_precision.py` | Single-image, single-class all-points interpolated AP |
| `detection_demo.py` | Four predictions: duplicate IoU 0.818182; AP 0.750000 → 0.833333 |
| `tiny_unet.py`, `unet_shapes.py` | Padded U-Net and feature-map shape inspection |
| `vision_shapes.py` | Original noisy circles/rectangles and paired flips |
| `segmentation_metrics.py`, `segment_shapes.py` | Loss, validation thresholds and three-seed comparison |
| `tiny_vit.py`, `vit_digits.py` | Sixteen patches plus class token; digit classification |
| `contrastive_toy.py` | Symmetric contrastive learning on noisy synthetic vectors |
| `pretrained_detector.py` | Optional cached SSDLite interface check |

```bash
.venv/bin/python chapter-17-vision/detection_demo.py
.venv/bin/python chapter-17-vision/unet_shapes.py
.venv/bin/python chapter-17-vision/vit_digits.py
.venv/bin/python chapter-17-vision/contrastive_toy.py
.venv/bin/python chapter-17-vision/segment_shapes.py
.venv/bin/python -m pytest chapter-17-vision --ignore-glob='*/based-on/*'
```

Measured reference: ViT training cross-entropy 2.494265 → 0.046712,
test accuracy 0.935556 after 30 epochs. Contrastive objective on fixed
diagnostic views: 4.711509 → 1.897906 after 200 updates. The points are not
images, and the contrastive loss is not a downstream performance measure.
See `applied/RESULTS.md` for per-class segmentation results and limitations.
AP here is not the full COCO evaluator. NMS is greedy and operates on one class.
Empty ground-truth AP is defined as zero locally; benchmark conventions differ.

## Data and provenance

The UCI optical digits are bundled with scikit-learn, are licensed CC BY 4.0,
and require attribution: E. Alpaydin and C. Kaynak, Optical Recognition of
Handwritten Digits, UCI Machine Learning Repository,
[doi:10.24432/C50P49](https://doi.org/10.24432/C50P49).
Synthetic images, masks and vectors are original and covered by this repository's MIT licence.

## Optional pretrained interface

Install the chapter's exact optional pin from the PyTorch index that matches the
installed torch build (`cpu`, or `cu132` for the CUDA build of chapter 11); the
PyPI wheel of torchvision fails to import with the CPU build of torch
(`operator torchvision::nms does not exist`):

```bash
uv pip install -p .venv -r chapter-17-vision/requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python chapter-17-vision/pretrained_detector.py --download
.venv/bin/python chapter-17-vision/pretrained_detector.py
```

This is TorchVision 0.29.1 with PyTorch 2.14.1. Use compatible CPU/CUDA wheels
as explained in the environment chapter. TorchVision is BSD-licensed software;
its [official pretrained-model documentation](https://docs.pytorch.org/vision/stable/models.html)
also explains that training datasets have separate applicable terms. Check
weight provenance and intended use rather than assuming the code licence
settles every dataset permission. The checkpoint is not redistributed here.
It is official `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`:

- URL: `https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth`
- SHA-256: `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`
- Default cache: `data/vision-ssdlite/`; override with `--cache PATH`.

There are no implicit downloads. Routine optional tests skip if the dependency
or checkpoint is absent; `NNBOOK_VISION_CACHE` chooses an existing test cache.
The original synthetic rectangle checks finite outputs and schema only.
It supplies no labelled photographic detection benchmark.

The guided exercise is in `applied/`: three incomplete functions in `based-on/`,
acceptance tests, a complete solution and measured JSON evidence in `full-code/`.
