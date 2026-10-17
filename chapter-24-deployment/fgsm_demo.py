"""Adversarial examples by the fast gradient sign method (FGSM)."""

import torch

from digit_model import load_images, trained_model


def fgsm(model, x, y, epsilon):
    """x + epsilon * sign(grad_x loss), kept in the valid pixel range, eq. (24.4)."""
    x = x.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    return (x + epsilon * x.grad.sign()).clamp(0, 1).detach()


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = trained_model()
    _, _, X_test, y_test = load_images()
    gen = torch.Generator().manual_seed(0)
    for epsilon in (0.0, 0.05, 0.1, 0.2):
        adversarial = fgsm(model, X_test, y_test, epsilon)
        noisy = (X_test + epsilon * torch.randint(0, 2, X_test.shape, generator=gen)
                 .mul(2).sub(1)).clamp(0, 1)                # random signs, same size
        with torch.inference_mode():
            acc_adv = (model(adversarial).argmax(1) == y_test).float().mean().item()
            acc_rnd = (model(noisy).argmax(1) == y_test).float().mean().item()
        print(f'epsilon {epsilon:.2f}: accuracy {acc_adv:.4f} under attack, '
              f'{acc_rnd:.4f} with random noise of the same size')
