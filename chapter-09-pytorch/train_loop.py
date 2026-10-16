"""Chapter 9: a complete PyTorch training loop on the digits of chapters 5 to 8.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn.
"""

from pathlib import Path

import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnbook.preprocessing import StandardScaler, train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
scaler = StandardScaler().fit(X[train])
to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
train_set = TensorDataset(to_tensor(scaler.transform(X[train])), torch.tensor(y[train]))
X_test, y_test = to_tensor(scaler.transform(X[test])), torch.tensor(y[test])
loader = DataLoader(train_set, batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(0))

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
loss_fn = nn.CrossEntropyLoss()                            # softmax + cross-entropy on scores
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()                              # gradients accumulate otherwise
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.inference_mode():
    accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()

if __name__ == "__main__":
    print(f"test accuracy after 20 epochs: {accuracy:.4f}")
    path = Path("outputs") / "digits_mlp.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)                   # save the parameters only
    copy = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    copy.load_state_dict(torch.load(path))                 # weights_only=True by default
    copy.eval()
    with torch.inference_mode():
        same = torch.equal(copy(X_test), model(X_test))
    print("reloaded model gives identical outputs:", same)
