"""A tiny character model on an author-written toy corpus; CPU only."""

import torch
from torch import nn

TEXT = 'rain falls. sun shines. wind blows. ' * 30


class CharacterModel(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 8)
        self.recurrent = nn.RNN(8, 16, batch_first=True)
        self.output = nn.Linear(16, vocabulary)

    def forward(self, tokens):
        hidden, _ = self.recurrent(self.embedding(tokens))
        return self.output(hidden)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    alphabet = sorted(set(TEXT))
    ids = torch.tensor([alphabet.index(c) for c in TEXT])
    length = 24
    starts = torch.arange(0, len(ids) - length - 1, length)
    x = torch.stack([ids[s:s+length] for s in starts])
    y = torch.stack([ids[s+1:s+length+1] for s in starts])
    model = CharacterModel(len(alphabet))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x).reshape(-1, len(alphabet)), y.reshape(-1))
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    generated = [alphabet.index('r')]
    with torch.inference_mode():
        final = nn.functional.cross_entropy(model(x).reshape(-1, len(alphabet)), y.reshape(-1)).item()
        for _ in range(47):
            logits = model(torch.tensor([generated]))[0, -1]
            generated.append(logits.argmax().item())
    return losses[0], final, ''.join(alphabet[i] for i in generated)


if __name__ == '__main__':
    initial, final, generated = run()
    print(f'initial training cross-entropy: {initial:.6f}')
    print(f'final training cross-entropy: {final:.6f}')
    print(f'greedy sample: {generated}')
