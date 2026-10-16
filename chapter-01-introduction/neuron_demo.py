"""Chapter 1: one neuron decides whether to irrigate a field."""

import numpy as np

# Features: soil moisture (0 dry .. 1 wet), temperature / 40 degrees C, chance of rain.
w = np.array([-3.0, 2.0, -2.5])
b = 1.0
days = {
    "dry and hot":         np.array([0.2, 0.75, 0.1]),
    "wet":                 np.array([0.8, 0.50, 0.6]),
    "dry, rain expected":  np.array([0.3, 0.60, 0.9]),
}

if __name__ == "__main__":
    for name, x in days.items():
        z = w @ x + b                          # weighted sum plus bias
        p = 1 / (1 + np.exp(-z))               # sigmoid activation
        decision = "irrigate" if z > 0 else "do not irrigate"
        print(f"{name:20s} z = {z:+.2f}, sigmoid = {p:.3f} -> {decision}")
