"""Fit an original repeated toy corpus; this measures memorisation only."""

import torch
from torch.nn import functional as F

from tiny_transformer import CharacterTransformer

TEXT = 'rain falls. sun shines. wind blows. '


def train(steps=200):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    alphabet = sorted(set(TEXT))
    ids = torch.tensor([alphabet.index(c) for c in TEXT * 4])
    x = torch.stack([ids[t:t + 16] for t in range(len(ids) - 16)])
    y = torch.stack([ids[t + 1:t + 17] for t in range(len(ids) - 16)])
    model = CharacterTransformer(len(alphabet))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.inference_mode():
        final = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        generated = [alphabet.index('r')]
        for _ in range(47):
            prefix = torch.tensor([generated[-model.context:]])
            generated.append(model(prefix)[0, -1].argmax().item())
    return initial, final, ''.join(alphabet[i] for i in generated)


if __name__ == '__main__':
    initial, final, sample = train()
    print(f'initial cross-entropy: {initial:.6f}')
    print(f'final cross-entropy: {final:.6f}')
    print(f'greedy sample: {sample}')
