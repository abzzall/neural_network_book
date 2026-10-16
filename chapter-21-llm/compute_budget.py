"""Compute-optimal model size from the fitted loss of Hoffmann et al. (2022).

L(N, D) = E + A / N**alpha + B / D**beta, with training compute C = 6 N D.
Constants from their Appendix D.2: E = 1.69, A = 406.4, B = 410.7,
alpha = 0.34, beta = 0.28.
"""

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(n_params, n_tokens):
    """Predicted final training loss (nats per token), eq. (21.5)."""
    return E + A / n_params ** ALPHA + B / n_tokens ** BETA


def optimal(compute):
    """Minimise loss(N, D) subject to 6 N D = compute (closed form)."""
    g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    n = g * (compute / 6) ** (BETA / (ALPHA + BETA))
    return n, compute / (6 * n)


if __name__ == '__main__':
    gopher = 6 * 280e9 * 300e9                  # 280B parameters, 300B tokens
    chinchilla = 6 * 70e9 * 1.4e12              # 70B parameters, 1.4T tokens
    for name, n, d in (('Gopher', 280e9, 300e9), ('Chinchilla', 70e9, 1.4e12)):
        print(f'{name}: compute {6 * n * d:.2e} FLOPs, predicted loss {loss(n, d):.3f}')
    n, d = optimal(gopher)
    print(f'optimum for the Gopher budget: {n / 1e9:.1f}B parameters, '
          f'{d / 1e12:.2f}T tokens, {d / n:.0f} tokens per parameter, '
          f'loss {loss(n, d):.3f}')
    print(f'Chinchilla as trained: {1.4e12 / 70e9:.0f} tokens per parameter')
    exponent = BETA / (ALPHA + BETA)
    print(f'size exponent beta/(alpha+beta) = {exponent:.4f}; '
          f'with 0.46 instead, the Gopher optimum scales by '
          f'{(gopher / 6) ** (0.46 - exponent):.2f}')
    for budget in (1e19, 1e21, 1e23):
        n, d = optimal(budget)
        print(f'budget {budget:.0e}: {n / 1e9:7.2f}B parameters, {d / 1e9:9.1f}B tokens')
