"""Train REINFORCE on the cart-pole task for three seeds and report its learning."""

import numpy as np
import torch

from reinforce import greedy, train
from rl_utils import evaluate, solved_at

torch.set_num_threads(1)
for seed in (0, 1, 2):
    history, policy = train(seed)
    print(f'seed {seed}: solved at episode {solved_at(history)}, '
          f'mean of last 50 episodes {np.mean(history[-50:]):.1f}, '
          f'greedy policy {evaluate(greedy(policy), seed):.1f}, '
          f'environment steps {int(sum(history))}')
