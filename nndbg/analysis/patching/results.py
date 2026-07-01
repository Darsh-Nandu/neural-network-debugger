"""Result type for the activation-patching / causal-tracing plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchingResult:
    """Patching result matrix: either logit-recovery (causal_trace) or
    logit-drop (mean_ablation).

    For ``method="causal_trace"``: ``matrix[i, j]`` is the fraction of the
    clean-vs-corrupted logit gap recovered by patching layer i at position j
    with the clean run's activation.  1.0 = full recovery, 0.0 = no effect.

    For ``method="mean_ablation"``: ``matrix[i, j]`` is
    ``clean_logit - ablated_logit`` when layer i, position j is replaced by
    the dataset mean.  Positive = that position matters; near zero = unimportant.
    """

    matrix: torch.Tensor   # (n_layers, n_positions)
    layers: list[str]
    positions: list[int]
    tokens: list[str] | None
    target: int
    method: str = "causal_trace"

    def best_cell(self) -> tuple[str, int, float]:
        """The (layer, position) with the highest absolute value."""
        abs_mat = self.matrix.abs()
        i, j = divmod(int(abs_mat.argmax()), abs_mat.shape[1])
        return self.layers[i], self.positions[j], float(self.matrix[i, j])

    def plot(self, ax=None):
        from nndbg.viz.plotting import heatmap

        pos_labels = [self.tokens[p] if self.tokens else str(p) for p in self.positions]
        is_ablation = self.method == "mean_ablation"
        title = (
            f"Mean ablation — logit drop (target id={self.target})"
            if is_ablation
            else f"Causal trace — logit recovery (target id={self.target})"
        )
        colorbar_label = "logit drop" if is_ablation else "recovery"
        return heatmap(
            self.matrix.numpy(),
            xticklabels=pos_labels,
            yticklabels=self.layers,
            title=title,
            xlabel="position",
            ylabel="layer",
            colorbar_label=colorbar_label,
            diverging=is_ablation,
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import heatmap

        pos_labels = [self.tokens[p] if self.tokens else str(p) for p in self.positions]
        title = (
            f"Mean ablation (target id={self.target})"
            if self.method == "mean_ablation"
            else f"Causal trace (target id={self.target})"
        )
        return heatmap(
            self.matrix.numpy(),
            xticklabels=pos_labels,
            yticklabels=self.layers,
            title=title,
        )

    def __repr__(self) -> str:
        layer, pos, score = self.best_cell()
        return (
            f"PatchingResult(method={self.method!r}, shape={tuple(self.matrix.shape)}, "
            f"target={self.target}, best=({layer!r}, pos={pos}, score={score:.3f}))"
        )
