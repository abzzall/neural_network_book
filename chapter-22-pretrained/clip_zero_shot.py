"""Zero-shot classification of generated shape images with CLIP: no training at all."""

from collections import Counter
import itertools

import numpy as np
import torch
from PIL import Image, ImageDraw

from clip_model import load_clip

SHAPES, COLOURS = ('circle', 'square', 'triangle'), ('red', 'green', 'blue')
RGB = {'red': (220, 30, 30), 'green': (30, 160, 50), 'blue': (30, 60, 220)}


def draw(shape, colour, rng, size=224):
    """A white image with one filled shape at a random position and size."""
    image = Image.new('RGB', (size, size), 'white')
    r = int(rng.integers(35, 80))
    cx, cy = (int(v) for v in rng.integers(r + 5, size - r - 5, size=2))
    box = [cx - r, cy - r, cx + r, cy + r]
    pen = ImageDraw.Draw(image)
    if shape == 'circle':
        pen.ellipse(box, fill=RGB[colour])
    elif shape == 'square':
        pen.rectangle(box, fill=RGB[colour])
    else:
        pen.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=RGB[colour])
    return image


@torch.inference_mode()
def zero_shot(processor, model, images, texts):
    """Index of the text whose embedding is most similar to each image's, eq. (22.4)."""
    inputs = processor(text=texts, images=images, return_tensors='pt', padding=True)
    return model(**inputs).logits_per_image.argmax(dim=1).numpy()


if __name__ == '__main__':
    torch.set_num_threads(4)
    processor, model = load_clip()
    rng = np.random.default_rng(0)
    classes = list(itertools.product(COLOURS, SHAPES))
    truth = [k for k in range(len(classes)) for _ in range(20)]
    images = [draw(classes[k][1], classes[k][0], rng) for k in truth]
    truth = np.array(truth)
    for template in ('{} {}', 'a photo of a {} {}.'):
        texts = [template.format(c, s) for c, s in classes]
        pred = zero_shot(processor, model, images, texts)
        shape_ok = np.mean([classes[p][1] == classes[t][1] for p, t in zip(pred, truth)])
        colour_ok = np.mean([classes[p][0] == classes[t][0] for p, t in zip(pred, truth)])
        print(f'prompt "{template.format("<colour>", "<shape>")}": both {np.mean(pred == truth):.3f}, '
              f'shape {shape_ok:.3f}, colour {colour_ok:.3f} ({len(truth)} images, 9 classes)')
        errors = Counter((' '.join(classes[t]), ' '.join(classes[q]))
                         for t, q in zip(truth, pred) if t != q)
        for (true, said), n in errors.most_common():
            print(f'  {n} x {true} -> {said}')
