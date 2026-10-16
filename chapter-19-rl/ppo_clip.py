"""PPO's clipped objective for one action, as a function of the probability ratio."""

import torch


def clipped_objective(ratio, advantage, epsilon=0.2):
    """min(r A, clip(r, 1 - eps, 1 + eps) A), eq. (19.9); PPO maximises its mean."""
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return torch.minimum(ratio * advantage, clipped * advantage)


if __name__ == '__main__':
    for advantage in (1.0, -1.0):
        print(f'advantage {advantage:+.0f}:')
        for r in (0.5, 0.9, 1.0, 1.1, 1.5):
            ratio = torch.tensor(r, requires_grad=True)
            value = clipped_objective(ratio, torch.tensor(advantage))
            value.backward()
            print(f'  ratio {r:.1f}: objective {value.item():+.2f}, '
                  f'gradient {ratio.grad.item():+.0f}')
