"""Result type for the concept-erasure (INLP) plane."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ErasureResult:
    """Outcome of iterative null-space projection (INLP) concept erasure.

    ``projection`` is a ``(d, d)`` matrix P such that ``X @ P.T`` removes
    the linearly-decodable signal for the target concept from a ``(n, d)``
    activation matrix. ``accuracy_trace`` records the probe accuracy *before*
    each projection step, so you can see how quickly the concept was erased.
    """

    concept: str
    layer: str
    projection: torch.Tensor      # (d, d) accumulated null-space projection
    accuracy_trace: list[float]   # probe train-accuracy before each iteration
    n_iters: int                  # iterations actually executed

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        """Project new activations through the learned erasure.

        Args:
            activations: ``(n, d)`` tensor with the same hidden dimension as
                the layer this erasure was trained on.
        Returns:
            ``(n, d)`` tensor with the concept direction(s) removed.
        """
        P = self.projection.float().to(activations.device)
        return activations.float() @ P.T

    def plot(self, ax=None):
        from nndbg.viz.plotting import line

        return line(
            list(range(1, len(self.accuracy_trace) + 1)),
            {"probe accuracy": self.accuracy_trace},
            title=f"INLP concept erasure — {self.concept!r} @ {self.layer}",
            xlabel="iteration",
            ylabel="probe train accuracy",
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import line

        return line(
            list(range(1, len(self.accuracy_trace) + 1)),
            {"probe accuracy": self.accuracy_trace},
            title=f"INLP concept erasure — {self.concept!r} @ {self.layer}",
            xlabel="iteration",
            ylabel="probe train accuracy",
        )

    def __repr__(self) -> str:
        final = self.accuracy_trace[-1] if self.accuracy_trace else float("nan")
        initial = self.accuracy_trace[0] if self.accuracy_trace else float("nan")
        return (
            f"ErasureResult(concept={self.concept!r}, layer={self.layer!r}, "
            f"iters={self.n_iters}, accuracy={initial:.3f} -> {final:.3f})"
        )
