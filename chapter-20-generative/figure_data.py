"""Write ring samples for the chapter's scatter plot: real data, GAN (seed 1), diffusion.

Usage: figure_data.py OUTPUT_FOLDER. Writes real.dat, gan.dat and diffusion.dat
(600 points each) and prints which modes each generator covered.
"""

from pathlib import Path
import sys

import numpy as np
import torch

import diffusion_2d
import gan_2d
from ring_data import mode_report, sample_ring


def write(folder, n=600):
    torch.set_num_threads(4)
    sets = {'real': sample_ring(n, np.random.default_rng(7)),
            'gan': gan_2d.train(1)[:n],                      # the seed that missed a mode
            'diffusion': diffusion_2d.generate(diffusion_2d.train(0), n)}
    for name, points in sets.items():
        np.savetxt(folder / f'{name}.dat', points, fmt='%.4f', header='x y', comments='')
    return {name: mode_report(points)[0] for name, points in sets.items()}


if __name__ == '__main__':
    print('modes covered:', write(Path(sys.argv[1])))
