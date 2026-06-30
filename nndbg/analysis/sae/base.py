"""SAEAnalyzer — what sparse, monosemantic features does this layer learn?

Implements a minimal trainable sparse autoencoder for decomposing a layer's
activations into sparse features, following Anthropic's "Towards
Monosemanticity" (Bricken et al., 2023).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nndbg.analysis.sae.results import SAEResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class SparseAutoencoder(nn.Module):
    """One hidden layer, overcomplete: ``hidden_dim -> n_features -> hidden_dim``.

    Trained with MSE reconstruction loss + an L1 penalty on the feature
    activations to encourage sparsity (most features off for any given
    input); decoder columns are re-normalized to unit norm after every
    step, the standard trick that keeps the L1 penalty from being
    trivially gamed by shrinking the decoder weights.
    """

    def __init__(self, input_dim: int, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return features, reconstruction

    def loss(self, x, features, reconstruction, l1_coeff: float = 1e-3) -> torch.Tensor:
        mse = F.mse_loss(reconstruction, x)
        l1 = l1_coeff * features.abs().mean()
        return mse + l1

    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class SAEAnalyzer:
    """Example::

        sae = inspector.sae.train(dataset, layer="transformer.h.4", n_features=512)
        result = inspector.sae.decompose(dataset, layer="transformer.h.4")
        result.plot()
    """

    def __init__(self, inspector: "Inspector") -> None:
        self._inspector = inspector
        self._trained: dict[str, SparseAutoencoder] = {}

    def train(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        n_features: int = 512,
        pooling: str = "last",
        epochs: int = 20,
        lr: float = 1e-3,
        l1_coeff: float = 1e-3,
        batch_size: int = 32,
    ) -> SparseAutoencoder:
        """Train a sparse autoencoder on ``layer``'s activations, collected
        across every example in ``dataset``. Registers the trained SAE so a
        later ``decompose(..., layer=layer)`` call can reuse it."""
        from torch.utils.data import DataLoader, TensorDataset

        inspector = self._inspector
        activations = collect_activations(
            inspector.model, inspector._hooks, dataset, [layer], pooling=pooling, device=inspector.device
        )
        X = activations[layer].float()
        input_dim = X.shape[-1]

        sae = SparseAutoencoder(input_dim, n_features).to(inspector.device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X), batch_size=min(batch_size, len(X)), shuffle=True)

        loss_history: list[float] = []
        sae.train()
        for _ in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(inspector.device)
                features, reconstruction = sae(batch)
                loss = sae.loss(batch, features, reconstruction, l1_coeff)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sae.renormalize_decoder()
                total_loss += loss.item()
            loss_history.append(total_loss / len(loader))

        sae.eval()
        sae.loss_history = loss_history  # stashed for SAEResult.plot_training()
        self._trained[layer] = sae
        return sae

    def decompose(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        sae: SparseAutoencoder | None = None,
        pooling: str = "last",
    ) -> SAEResult:
        """Run ``dataset`` through a (trained) SAE and report per-example
        feature activations, reconstruction quality, and sparsity."""
        if sae is None:
            if layer not in self._trained:
                raise RuntimeError(
                    f"No trained SAE for layer {layer!r}. Call "
                    f"inspector.sae.train(dataset, layer={layer!r}) first, or pass sae=..."
                )
            sae = self._trained[layer]

        inspector = self._inspector
        activations = collect_activations(
            inspector.model, inspector._hooks, dataset, [layer], pooling=pooling, device=inspector.device
        )
        X = activations[layer].float().to(inspector.device)

        sae.eval()
        with torch.no_grad():
            features, reconstruction = sae(X)

        recon_loss = F.mse_loss(reconstruction, X).item()
        sparsity = (features.abs() < 1e-6).float().mean().item()

        return SAEResult(
            feature_activations=features.cpu(),
            reconstruction=reconstruction.cpu(),
            reconstruction_loss=recon_loss,
            sparsity=sparsity,
            layer=layer,
            loss_history=list(getattr(sae, "loss_history", [])),
        )
