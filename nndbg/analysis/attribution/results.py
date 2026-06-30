"""Result type for the attribution analysis plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttributionResult:
    """Per-position importance scores explaining a model's output."""

    method: str
    scores: torch.Tensor  # 1D: one score per input position/feature
    tokens: list[str] | None = None
    layer: str | None = None

    def top_k(self, k: int = 5) -> list[tuple[int, float]]:
        """The ``k`` highest-scoring positions as ``(index, score)`` pairs."""
        k = min(k, len(self.scores))
        vals, idx = self.scores.topk(k)
        return list(zip(idx.tolist(), vals.tolist()))

    def _labels(self) -> list[str]:
        if self.tokens is not None:
            return self.tokens
        return [str(i) for i in range(len(self.scores))]

    def plot(self, ax=None):
        from nndbg.viz.plotting import bar

        title = f"{self.method} attribution" + (f" @ {self.layer}" if self.layer else "")
        return bar(self._labels(), self.scores.numpy(), title=title, xlabel="position", ylabel="importance", ax=ax)

    def plotly(self):
        from nndbg.viz.plotly_backend import bar

        title = f"{self.method} attribution" + (f" @ {self.layer}" if self.layer else "")
        return bar(self._labels(), self.scores.tolist(), title=title)

    def __repr__(self) -> str:
        return f"AttributionResult(method={self.method!r}, n={len(self.scores)}, top3={self.top_k(3)})"
