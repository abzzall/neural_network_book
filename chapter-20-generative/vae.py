"""A variational autoencoder for 8x8 digits with a two-dimensional latent space."""

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.mean, self.log_var = nn.Linear(64, latent), nn.Linear(64, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.mean(h), self.log_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)   # reparameterisation
        return self.decoder(z), mu, log_var                         # decoder gives logits


def loss_terms(model, x):
    """Per-image reconstruction loss and KL term; their sum is the negative ELBO."""
    logits, mu, log_var = model(x)
    reconstruction = nn.functional.binary_cross_entropy_with_logits(
        logits, x, reduction='none').sum(dim=1)                    # Bernoulli pixels
    kl = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(dim=1)  # eq. (20.4)
    return reconstruction.mean(), kl.mean()


def train(images, latent=2, seed=0, epochs=200, batch=64, lr=1e-3):
    torch.manual_seed(seed)
    model = VAE(latent)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(images, dtype=torch.float32)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=order).split(batch):
            reconstruction, kl = loss_terms(model, x[idx])
            optimiser.zero_grad()
            (reconstruction + kl).backward()                        # minus the ELBO
            optimiser.step()
    return model.eval()


def sample(model, n, seed=0):
    """New images: decode latent points drawn from the prior N(0, I)."""
    z = torch.randn(n, model.mean.out_features, generator=torch.Generator().manual_seed(seed))
    with torch.inference_mode():
        return torch.sigmoid(model.decoder(z)).numpy()
