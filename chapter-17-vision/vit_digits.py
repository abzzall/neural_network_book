"""A tiny ViT trained from scratch on the bundled UCI digits (CC BY 4.0)."""

import torch
from sklearn.datasets import load_digits

from nnbook.preprocessing import train_test_split
from tiny_vit import TinyViT


def run(epochs=30):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=.25, seed=0, stratify=y)
    images = torch.tensor(x / 16., dtype=torch.float32).reshape(-1, 1, 8, 8)
    labels = torch.tensor(y)
    model = TinyViT()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    generator = torch.Generator().manual_seed(42)
    initial = torch.nn.functional.cross_entropy(model(images[train]), labels[train]).item()
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(train), generator=generator).split(64):
            ids = train[batch]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images[ids]), labels[ids])
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.inference_mode():
        final = torch.nn.functional.cross_entropy(model(images[train]), labels[train]).item()
        accuracy = (model(images[test]).argmax(-1) == labels[test]).float().mean().item()
    return initial, final, accuracy


if __name__ == '__main__':
    initial, final, accuracy = run()
    print('patches plus class token: 17')
    print(f'initial training cross-entropy: {initial:.6f}')
    print(f'final training cross-entropy: {final:.6f}')
    print(f'test accuracy: {accuracy:.6f}')
