"""Shared matplotlib helpers used by every analysis plane's Result.plot()."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nndbg.viz.style import DIVERGING_CMAP, DPI, FIGSIZE, HEATMAP_CMAP, PALETTE, apply_style


def _new_ax(ax, figsize=FIGSIZE):
    if ax is not None:
        return ax.figure, ax
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    return fig, ax


def heatmap(
    matrix: np.ndarray,
    *,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str | None = None,
    diverging: bool = False,
    colorbar_label: str | None = None,
    ax=None,
):
    """2D heatmap — attention matrices, patching recovery grids, SAE
    activation grids, attribution-over-positions maps."""
    fig, ax = _new_ax(ax)
    matrix = np.asarray(matrix)

    if diverging:
        vmax = np.abs(matrix).max() or 1.0
        im = ax.imshow(matrix, cmap=cmap or DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(matrix, cmap=cmap or HEATMAP_CMAP, aspect="auto")

    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_style(ax)
    fig.tight_layout()
    return ax


def line(
    x: Sequence,
    series: dict[str, Sequence[float]] | Sequence[float],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    yerr: dict[str, Sequence[float]] | Sequence[float] | None = None,
    ax=None,
):
    """Line plot — probe accuracy-by-layer, SAE/VAE training loss curves."""
    fig, ax = _new_ax(ax)

    if not isinstance(series, dict):
        series = {"": series}
        yerr = {"": yerr} if yerr is not None else None

    for i, (name, ys) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        err = yerr.get(name) if isinstance(yerr, dict) else None
        if err is not None:
            ax.errorbar(x, ys, yerr=err, label=name or None, color=color, marker="o", capsize=3)
        else:
            ax.plot(x, ys, label=name or None, color=color, marker="o")

    if any(name for name in series):
        ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_style(ax)
    fig.tight_layout()
    return ax


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    labels: Sequence | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
):
    """2D scatter — VAE/PCA latent space, geometry projections, optionally
    colored by a categorical or continuous ``labels`` array."""
    fig, ax = _new_ax(ax)
    x = np.asarray(x)
    y = np.asarray(y)

    if labels is None:
        ax.scatter(x, y, color=PALETTE[0], alpha=0.8, edgecolors="white", linewidths=0.3)
    else:
        labels = np.asarray(labels)
        if np.issubdtype(labels.dtype, np.number) and len(np.unique(labels)) > len(PALETTE):
            sc = ax.scatter(x, y, c=labels, cmap=HEATMAP_CMAP, alpha=0.85, edgecolors="white", linewidths=0.3)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            for i, value in enumerate(sorted(set(labels.tolist()))):
                mask = labels == value
                ax.scatter(
                    x[mask], y[mask],
                    label=str(value), color=PALETTE[i % len(PALETTE)],
                    alpha=0.85, edgecolors="white", linewidths=0.3,
                )
            ax.legend(fontsize=8)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "dim 1")
    ax.set_ylabel(ylabel or "dim 2")
    apply_style(ax)
    fig.tight_layout()
    return ax


def bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
):
    """Bar chart — SAE top-feature activations, neuron/head importance."""
    fig, ax = _new_ax(ax)
    ax.bar(range(len(labels)), values, color=PALETTE[1])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_style(ax)
    fig.tight_layout()
    return ax
