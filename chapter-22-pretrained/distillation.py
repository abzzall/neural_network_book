"""Knowledge distillation: a 16-unit student learns from a 256-unit teacher's soft outputs."""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

T = 4.0                                              # softening temperature


def mlp(hidden):
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def distillation_loss(student_logits, teacher_logits, temperature=T):
    """T^2 KL(softmax(teacher/T) || softmax(student/T)), eq. (22.3)."""
    return temperature ** 2 * nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1), reduction='batchmean')


def fit(model, loss, x_test, y_test, epochs=300):
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):                            # full-batch training
        value = loss(model)
        optimiser.zero_grad()
        value.backward()
        optimiser.step()
    with torch.no_grad():
        return float((model(x_test).argmax(1).numpy() == y_test).mean())


if __name__ == '__main__':
    torch.set_num_threads(4)
    images, labels = load_digits(return_X_y=True)
    tr, te, y_tr, y_te = train_test_split(images / 16, labels, test_size=0.3,
                                          random_state=0, stratify=labels)
    x, x_te, y = (torch.tensor(a, dtype=torch.float32) for a in (tr, te, y_tr))
    y = y.long()
    few = np.random.default_rng(0).choice(len(tr), 100, replace=False)
    ce = nn.functional.cross_entropy
    size = lambda m: sum(p.numel() for p in m.parameters())
    print(f'parameters: teacher {size(mlp(256))}, student {size(mlp(16))}')
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        teacher = mlp(256)
        acc = {'teacher': fit(teacher, lambda m: ce(m(x), y), x_te, y_te)}
        with torch.no_grad():
            soft = teacher(x)
        for name, loss in (('100 labels', lambda m: ce(m(x[few]), y[few])),
                           ('all labels', lambda m: ce(m(x), y)),
                           ('distilled, no labels', lambda m: distillation_loss(m(x), soft))):
            torch.manual_seed(seed)
            acc[name] = fit(mlp(16), loss, x_te, y_te)
        print(f'seed {seed}: ' + ', '.join(f'{k} {v:.4f}' for k, v in acc.items()))
