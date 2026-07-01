"""GeometryAnalyzer — how similar are a model's layers to each other or to another model?

Implements linear CKA (Kornblith et al. 2019) as the primary similarity
metric: mathematically sound, scale-invariant, and fast for the feature
dimensions typical in modern networks.  Also exposes PCA (and optional
UMAP) projections for visualising a single layer's representation space.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from nndbg.analysis.geometry.results import GeometryResult, ProjectionResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between centered activation matrices (n, d1) and (n, d2).

    Uses the matrix-multiplication form: HSIC_linear(X,Y) = ||X'Y||_F^2
    This is O(n*d^2) and avoids forming the n*n Gram matrices.
    """
    X = (X - X.mean(0)).float()
    Y = (Y - Y.mean(0)).float()
    XtY = X.T @ Y
    hsic_xy = float((XtY * XtY).sum())
    XtX = X.T @ X
    hsic_xx = float((XtX * XtX).sum())
    YtY = Y.T @ Y
    hsic_yy = float((YtY * YtY).sum())
    denom = (hsic_xx * hsic_yy) ** 0.5
    return hsic_xy / denom if denom > 1e-10 else 0.0


class GeometryAnalyzer:
    """Representational geometry analysis: CKA similarity and PCA projections.

    Example — self-similarity (is the network learning new representations
    layer-by-layer, or do layers converge?)::

        result = inspector.geometry.layer_similarity(dataset, layers=inspector.find_layers(r"h\\.\\d+$"))
        result.plot()

    Example — cross-model comparison (how much did fine-tuning change each layer?)::

        result = inspector.geometry.compare(finetuned_inspector, dataset)
        result.plot()

    Example — 2-D PCA scatter::

        result = inspector.geometry.pca(dataset, layer="transformer.h.5", labels=class_labels)
        result.plot()
    """

    def __init__(self, inspector: Inspector) -> None:
        self._inspector = inspector

    # ------------------------------------------------------------------
    def layer_similarity(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layers: list[str] | None = None,
        pooling: str = "last",
    ) -> GeometryResult:
        """Compute pairwise linear CKA between every pair of layers.

        Args:
            dataset: a list of input tensors (same format the model accepts).
            layers: subset of layer names to compare. Defaults to all
                registered layers — for large models, pass a one-per-block
                subset (e.g. ``inspector.find_layers(r"h\\.\\d+$")``) to keep
                the ``O(L^2)`` computation tractable.
            pooling: how to reduce the sequence dimension for 3-D activations.
        Returns:
            GeometryResult whose ``similarity[i, j]`` is CKA(layer_i, layer_j).
        """
        layer_names = layers or self._inspector.layers()
        acts = self._collect(dataset, layer_names, pooling)
        n = len(layer_names)
        sim = torch.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                v = _linear_cka(acts[layer_names[i]], acts[layer_names[j]])
                sim[i, j] = v
                sim[j, i] = v
        return GeometryResult(similarity=sim, row_layers=layer_names, col_layers=layer_names)

    def compare(
        self,
        other: Inspector,
        dataset: Sequence[torch.Tensor],
        *,
        layers: list[str] | None = None,
        other_layers: list[str] | None = None,
        pooling: str = "last",
    ) -> GeometryResult:
        """Cross-model CKA: compare this inspector's layers against another
        inspector's layers on the same dataset.

        Useful for checking how much fine-tuning (or any other training)
        changed each layer's representation. ``similarity[i, j]`` = CKA
        between ``layers[i]`` of this model and ``other_layers[j]`` of the
        other model.

        Args:
            other: a second Inspector wrapping a different model checkpoint.
            dataset: inputs compatible with BOTH models.
            layers: layers from this model (default: all).
            other_layers: layers from the other model (default: same as
                ``layers``).
        """
        layer_names = layers or self._inspector.layers()
        other_layer_names = other_layers or other.layers()

        acts_self = self._collect(dataset, layer_names, pooling)
        acts_other = _collect_other(other, dataset, other_layer_names, pooling)

        n_self = len(layer_names)
        n_other = len(other_layer_names)
        sim = torch.zeros(n_self, n_other)
        for i, ln_s in enumerate(layer_names):
            for j, ln_o in enumerate(other_layer_names):
                sim[i, j] = _linear_cka(acts_self[ln_s], acts_other[ln_o])

        return GeometryResult(similarity=sim, row_layers=layer_names, col_layers=other_layer_names)

    def pca(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        pooling: str = "last",
        n_components: int = 2,
        labels: list | None = None,
    ) -> ProjectionResult:
        """Project a layer's activations to ``n_components`` dims via PCA.

        Args:
            layer: layer whose activations to project.
            n_components: number of principal components (2 for a scatter plot).
            labels: optional per-example labels for coloring the scatter.
        """
        from sklearn.decomposition import PCA

        acts = self._collect(dataset, [layer], pooling)
        X = acts[layer].float().numpy()
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        coords = pca.fit_transform(X)
        return ProjectionResult(
            coords=torch.from_numpy(coords),
            explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
            layer=layer,
            method="pca",
            labels=labels,
        )

    def umap(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        pooling: str = "last",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        labels: list | None = None,
    ) -> ProjectionResult:
        """Project via UMAP (requires ``pip install umap-learn``).

        UMAP preserves local neighbourhood structure better than PCA for
        large, high-dimensional datasets. Falls back to a helpful error if
        ``umap-learn`` is not installed.
        """
        try:
            import umap  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "umap-learn is required for GeometryAnalyzer.umap(). "
                "Install it with: pip install umap-learn"
            ) from None

        acts = self._collect(dataset, [layer], pooling)
        X = acts[layer].float().numpy()
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        coords = reducer.fit_transform(X)
        return ProjectionResult(
            coords=torch.from_numpy(coords),
            explained_variance_ratio=[],  # UMAP doesn't give explained variance
            layer=layer,
            method="umap",
            labels=labels,
        )

    # ------------------------------------------------------------------
    def _collect(self, dataset, layer_names, pooling):
        inspector = self._inspector
        return collect_activations(
            inspector.model, inspector._hooks, dataset, layer_names, pooling=pooling, device=inspector.device
        )


def _collect_other(other: Inspector, dataset, layer_names, pooling):
    return collect_activations(
        other.model, other._hooks, dataset, layer_names, pooling=pooling, device=other.device
    )
