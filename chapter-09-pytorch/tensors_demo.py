"""Chapter 9: tensors -- shapes, dtypes, broadcasting, and NumPy interchange."""

import numpy as np
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # shape (2, 3)
b = torch.tensor([10.0, 20.0, 30.0])                       # shape (3,)

if __name__ == "__main__":
    print("shape:", tuple(x.shape), " dtype:", x.dtype)
    print("x + b broadcasts b over the rows:", (x + b).tolist())
    print("x @ x.T =", (x @ x.T).tolist())
    a = np.array([1.0, 2.0])
    t = torch.from_numpy(a)                                # shares memory with a
    a[0] = 99.0
    print("NumPy default dtype:", a.dtype, "-> tensor dtype:", t.dtype, "first value:", t[0].item())
    print("float32 epsilon:", torch.finfo(torch.float32).eps)
