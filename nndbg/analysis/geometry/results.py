"""Result types for the geometry analysis plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GeometryResult:
    """Pairwise linear CKA similarity between layers (or between two models).

    ``similarity[i, j]`` is the CKA score between ``row_layers[i]`` and
    ``col_layers[j]``. Values near 1 = representations are linearly similar;
    near 0 = unrelated."""

    similarity: torch.Tensor   # (len(row_layers), len(col_layers))
    row_layers: list[str]
    col_layers: list[str]

    def plot(self, ax=None):
        from nndbg.viz.plotting import heatmap

        row_labels = [ln.split(".")[-2] + "." + ln.split(".")[-1] if "." in ln else ln for ln in self.row_layers]
        col_labels = [ln.split(".")[-2] + "." + ln.split(".")[-1] if "." in ln else ln for ln in self.col_layers]
        return heatmap(
            self.similarity.numpy(),
            xticklabels=col_labels,
            yticklabels=row_labels,
            title="Linear CKA similarity between layers",
            xlabel="layer",
            ylabel="layer",
            colorbar_label="CKA",
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import heatmap

        row_labels = [ln.split(".")[-1] for ln in self.row_layers]
        col_labels = [ln.split(".")[-1] for ln in self.col_layers]
        return heatmap(
            self.similarity.numpy(),
            xticklabels=col_labels,
            yticklabels=row_labels,
            title="Linear CKA similarity between layers",
        )

    def __repr__(self) -> str:
        diag = self.similarity.diagonal().mean().item() if self.similarity.shape[0] == self.similarity.shape[1] else None
        s = f"GeometryResult(shape={tuple(self.similarity.shape)}"
        if diag is not None:
            s += f", self_sim_diag={diag:.3f}"
        return s + ")"


@dataclass
class ProjectionResult:
    """PCA (or UMAP if installed) projection of a layer's activations."""

    coords: torch.Tensor               # (n_samples, n_components)
    explained_variance_ratio: list[float]
    layer: str
    method: str = "pca"
    labels: list | None = None

    def plot(self, ax=None):
        from nndbg.viz.plotting import scatter

        x = self.coords[:, 0].numpy()
        y = self.coords[:, 1].numpy() if self.coords.shape[1] > 1 else x * 0
        evr = self.explained_variance_ratio
        xlabel = f"dim 1 ({evr[0]:.1%})" if evr else "dim 1"
        ylabel = f"dim 2 ({evr[1]:.1%})" if len(evr) > 1 else "dim 2"
        return scatter(
            x,
            y,
            labels=self.labels,
            title=f"{self.method.upper()} projection — {self.layer}",
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import scatter

        x = self.coords[:, 0].tolist()
        y = self.coords[:, 1].tolist() if self.coords.shape[1] > 1 else [0.0] * len(x)
        return scatter(x, y, labels=self.labels, title=f"{self.method.upper()} projection — {self.layer}")

    def __repr__(self) -> str:
        evr = self.explained_variance_ratio
        total = sum(evr)
        return (
            f"ProjectionResult(layer={self.layer!r}, method={self.method!r}, "
            f"n={len(self.coords)}, variance_explained={total:.1%})"
        )
