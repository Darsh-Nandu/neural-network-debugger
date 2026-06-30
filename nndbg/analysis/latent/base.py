"""LatentAnalyzer — where do activations sit in a compressed latent space?

Trains a small VAE on a layer's pooled activations, giving you a
low-dimensional, visualizable coordinate for every example plus a
reconstruction-error signal for flagging out-of-distribution activations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nndbg.analysis.latent.results import LatentResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class ActivationVAE(nn.Module):
    """A small variational autoencoder over activation vectors:
    ``input_dim -> hidden_dim -> latent_dim`` (encoder, with a
    reparameterized Gaussian bottleneck) and the mirrored decoder.
    """

    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.encoder_hidden(x))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.decoder_hidden(z))
        return self.decoder_out(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss(self, x, reconstruction, mu, logvar, kl_weight: float = 1e-3) -> torch.Tensor:
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl


class LatentAnalyzer:
    """Example::

        vae = inspector.latent.train(dataset, layer="transformer.h.4", latent_dim=2)
        result = inspector.latent.encode(dataset, layer="transformer.h.4")
        result.plot()
        result.anomalies()  # indices of outlier examples
    """

    def __init__(self, inspector: "Inspector") -> None:
        self._inspector = inspector
        self._trained: dict[str, ActivationVAE] = {}

    def train(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        latent_dim: int = 2,
        hidden_dim: int = 64,
        pooling: str = "last",
        epochs: int = 50,
        lr: float = 1e-3,
        kl_weight: float = 1e-3,
        batch_size: int = 32,
    ) -> ActivationVAE:
        """Train a VAE on ``layer``'s activations, collected across every
        example in ``dataset``. Registers the trained VAE so a later
        ``encode(..., layer=layer)`` call can reuse it."""
        from torch.utils.data import DataLoader, TensorDataset

        inspector = self._inspector
        activations = collect_activations(
            inspector.model, inspector._hooks, dataset, [layer], pooling=pooling, device=inspector.device
        )
        X = activations[layer].float()
        input_dim = X.shape[-1]

        vae = ActivationVAE(input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(inspector.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X), batch_size=min(batch_size, len(X)), shuffle=True)

        loss_history: list[float] = []
        vae.train()
        for _ in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(inspector.device)
                reconstruction, mu, logvar = vae(batch)
                loss = vae.loss(batch, reconstruction, mu, logvar, kl_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss / len(loader))

        vae.eval()
        vae.loss_history = loss_history
        self._trained[layer] = vae
        return vae

    def encode(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        vae: ActivationVAE | None = None,
        pooling: str = "last",
        labels: list | None = None,
    ) -> LatentResult:
        """Encode ``dataset`` through a (trained) VAE: the deterministic
        latent mean per example, plus a reconstruction error usable as an
        anomaly score."""
        if vae is None:
            if layer not in self._trained:
                raise RuntimeError(
                    f"No trained VAE for layer {layer!r}. Call "
                    f"inspector.latent.train(dataset, layer={layer!r}) first, or pass vae=..."
                )
            vae = self._trained[layer]

        inspector = self._inspector
        activations = collect_activations(
            inspector.model, inspector._hooks, dataset, [layer], pooling=pooling, device=inspector.device
        )
        X = activations[layer].float().to(inspector.device)

        vae.eval()
        with torch.no_grad():
            mu, _ = vae.encode(X)
            reconstruction = vae.decode(mu)
            errors = F.mse_loss(reconstruction, X, reduction="none").mean(dim=-1)

        return LatentResult(
            latent=mu.cpu(),
            reconstruction_error=errors.cpu(),
            layer=layer,
            loss_history=list(getattr(vae, "loss_history", [])),
            labels=labels,
        )
