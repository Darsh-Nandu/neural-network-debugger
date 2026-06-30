"""Result type for the attention analysis plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionResult:
    """Attention weights for one layer (``kind="heads"``), a cumulative
    attention-flow matrix (``kind="rollout"``), or per-head entropy
    (``kind="entropy"``)."""

    kind: str
    matrix: torch.Tensor
    tokens: list[str] | None = None
    layer: int | None = None
    head_scores: torch.Tensor | None = None

    def plot(self, max_heads: int = 8):
        if self.kind == "heads":
            return self._plot_heads(max_heads)
        if self.kind == "rollout":
            return self._plot_rollout()
        if self.kind == "entropy":
            return self._plot_entropy()
        raise ValueError(f"Unknown AttentionResult.kind: {self.kind!r}")

    def _plot_heads(self, max_heads: int):
        import matplotlib.pyplot as plt

        from nndbg.viz.style import HEATMAP_CMAP

        n_heads = min(self.matrix.shape[0], max_heads)
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), dpi=100, squeeze=False)
        axes = axes.flatten()
        for i in range(n_heads):
            ax = axes[i]
            ax.imshow(self.matrix[i].numpy(), cmap=HEATMAP_CMAP, aspect="auto")
            ax.set_title(f"head {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for j in range(n_heads, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Attention heads — layer {self.layer}")
        fig.tight_layout()
        return axes[:n_heads]

    def _plot_rollout(self):
        from nndbg.viz.plotting import heatmap

        return heatmap(
            self.matrix.numpy(),
            xticklabels=self.tokens,
            yticklabels=self.tokens,
            title="Attention rollout (cumulative information flow)",
            xlabel="attends to",
            ylabel="from token",
        )

    def _plot_entropy(self):
        from nndbg.viz.plotting import heatmap

        n_layers, n_heads = self.head_scores.shape
        return heatmap(
            self.head_scores.numpy(),
            xticklabels=[f"head {h}" for h in range(n_heads)],
            yticklabels=[f"layer {l}" for l in range(n_layers)],
            title="Attention entropy per head (higher = more diffuse)",
            colorbar_label="entropy (nats)",
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import heatmap

        if self.kind == "heads":
            return heatmap(self.matrix[0].numpy(), title=f"Attention head 0 — layer {self.layer}")
        if self.kind == "rollout":
            return heatmap(
                self.matrix.numpy(), xticklabels=self.tokens, yticklabels=self.tokens, title="Attention rollout"
            )
        return heatmap(self.head_scores.numpy(), title="Attention entropy per head")

    def __repr__(self) -> str:
        return f"AttentionResult(kind={self.kind!r}, shape={tuple(self.matrix.shape)}, layer={self.layer})"
