"""Result type for the VAE latent-space analysis plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LatentResult:
    """A VAE's compressed view of a layer's activations: a low-dimensional
    latent coordinate plus a reconstruction error per example, which doubles
    as an anomaly/outlier score — examples the VAE struggles to reconstruct
    sit outside the distribution it learned."""

    latent: torch.Tensor              # (n_samples, latent_dim)
    reconstruction_error: torch.Tensor  # (n_samples,)
    layer: str
    loss_history: list[float]
    labels: list | None = None

    def anomalies(self, threshold: float | None = None) -> list[int]:
        """Indices of examples whose reconstruction error exceeds
        ``threshold`` (defaults to mean + 2 std of the observed errors)."""
        errors = self.reconstruction_error
        if threshold is None:
            threshold = float(errors.mean() + 2 * errors.std())
        return (errors > threshold).nonzero(as_tuple=True)[0].tolist()

    def _2d_coords(self):
        if self.latent.shape[1] == 2:
            return self.latent[:, 0].numpy(), self.latent[:, 1].numpy()
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2).fit_transform(self.latent.numpy())
        return coords[:, 0], coords[:, 1]

    def plot(self, *, color_by: str = "error", ax=None):
        from nndbg.viz.plotting import scatter

        x, y = self._2d_coords()
        if color_by == "label" and self.labels is not None:
            color_values = self.labels
        else:
            color_values = self.reconstruction_error.numpy()

        suffix = "" if self.latent.shape[1] == 2 else " (PCA-projected)"
        return scatter(
            x,
            y,
            labels=color_values,
            title=f"VAE latent space — {self.layer}{suffix}",
            xlabel="latent dim 1",
            ylabel="latent dim 2",
            ax=ax,
        )

    def plot_training(self, ax=None):
        from nndbg.viz.plotting import line

        return line(
            list(range(1, len(self.loss_history) + 1)),
            {"loss": self.loss_history},
            title=f"VAE training loss — {self.layer}",
            xlabel="epoch",
            ylabel="ELBO loss",
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import scatter

        x, y = self._2d_coords()
        return scatter(
            x, y, labels=self.reconstruction_error.tolist(), title=f"VAE latent space — {self.layer}"
        )

    def __repr__(self) -> str:
        n_anom = len(self.anomalies())
        return (
            f"LatentResult(layer={self.layer!r}, n={len(self.latent)}, "
            f"latent_dim={self.latent.shape[1]}, anomalies={n_anom})"
        )
