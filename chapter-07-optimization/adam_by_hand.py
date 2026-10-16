"""Chapter 7: three steps of Adam on f(t) = t^2 from t = 1, every quantity shown."""

import numpy as np

lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
t, m, v = 1.0, 0.0, 0.0

for step in (1, 2, 3):
    g = 2 * t                                   # gradient of t^2
    m = beta1 * m + (1 - beta1) * g             # running average of gradients
    v = beta2 * v + (1 - beta2) * g ** 2        # running average of squared gradients
    m_hat = m / (1 - beta1 ** step)             # bias corrections
    v_hat = v / (1 - beta2 ** step)
    t = t - lr * m_hat / (np.sqrt(v_hat) + eps)
    print(f"step {step}: g = {g:.4f}, m = {m:.4f}, v = {v:.6f}, "
          f"m_hat = {m_hat:.4f}, v_hat = {v_hat:.4f}, t = {t:.4f}")
