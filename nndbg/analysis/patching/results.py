"""Result type for the activation-patching / causal-tracing plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchingResult:
    """Logit-recovery matrix from a causal trace: how much, per (layer,
    position), restoring the clean run's activation pulls the corrupted
    run's target logit back toward the clean baseline. ``1.0`` = full
    recovery, ``0.0`` = no effect."""

    matrix: torch.Tensor  # (n_layers, n_positions)
    layers: list[str]
    positions: list[int]
    tokens: list[str] | None
    target: int

    def best_cell(self) -> tuple[str, int, float]:
        """The (layer, position) with the highest recovery."""
        i, j = divmod(int(self.matrix.argmax()), self.matrix.shape[1])
        return self.layers[i], self.positions[j], float(self.matrix[i, j])

    def plot(self, ax=None):
        from nndbg.viz.plotting import heatmap

        pos_labels = [self.tokens[p] if self.tokens else str(p) for p in self.positions]
        return heatmap(
            self.matrix.numpy(),
            xticklabels=pos_labels,
            yticklabels=self.layers,
            title=f"Causal trace — logit recovery (target id={self.target})",
            xlabel="position",
            ylabel="layer",
            colorbar_label="recovery",
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import heatmap

        pos_labels = [self.tokens[p] if self.tokens else str(p) for p in self.positions]
        return heatmap(
            self.matrix.numpy(),
            xticklabels=pos_labels,
            yticklabels=self.layers,
            title=f"Causal trace — logit recovery (target id={self.target})",
        )

    def __repr__(self) -> str:
        layer, pos, score = self.best_cell()
        return (
            f"PatchingResult(shape={tuple(self.matrix.shape)}, target={self.target}, "
            f"best=({layer!r}, pos={pos}, recovery={score:.3f}))"
        )
