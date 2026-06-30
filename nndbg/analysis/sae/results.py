"""Result type for the sparse-autoencoder analysis plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SAEResult:
    """Decomposition of a layer's activations into sparse features."""

    feature_activations: torch.Tensor  # (n_samples, n_features)
    reconstruction: torch.Tensor       # (n_samples, hidden_dim)
    reconstruction_loss: float
    sparsity: float                    # fraction of (sample, feature) entries that are ~0
    layer: str
    loss_history: list[float]

    @property
    def active_features(self) -> torch.Tensor:
        """Boolean mask of which features fired at least once."""
        return (self.feature_activations.abs() > 1e-6).any(dim=0)

    def top_features(self, k: int = 10) -> list[int]:
        """Indices of the ``k`` features with the highest mean activation."""
        mean_acts = self.feature_activations.abs().mean(dim=0)
        k = min(k, mean_acts.numel())
        return mean_acts.topk(k).indices.tolist()

    def plot(self, k: int = 20, ax=None):
        from nndbg.viz.plotting import bar

        top = self.top_features(k)
        vals = self.feature_activations[:, top].abs().mean(dim=0).numpy()
        return bar(
            [str(i) for i in top],
            vals,
            title=f"SAE top-{len(top)} features — {self.layer}",
            xlabel="feature index",
            ylabel="mean |activation|",
            ax=ax,
        )

    def plot_training(self, ax=None):
        from nndbg.viz.plotting import line

        return line(
            list(range(1, len(self.loss_history) + 1)),
            {"loss": self.loss_history},
            title=f"SAE training loss — {self.layer}",
            xlabel="epoch",
            ylabel="reconstruction + L1 loss",
            ax=ax,
        )

    def plotly(self, k: int = 20):
        from nndbg.viz.plotly_backend import bar

        top = self.top_features(k)
        vals = self.feature_activations[:, top].abs().mean(dim=0).tolist()
        return bar([str(i) for i in top], vals, title=f"SAE top-{len(top)} features — {self.layer}")

    def __repr__(self) -> str:
        n_active = int(self.active_features.sum())
        n_total = self.feature_activations.shape[1]
        return (
            f"SAEResult(layer={self.layer!r}, recon_loss={self.reconstruction_loss:.4f}, "
            f"sparsity={self.sparsity:.3f}, active_features={n_active}/{n_total})"
        )
