"""Result type for the probing analysis plane."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Cross-validated linear-probe accuracy for a concept, layer by layer."""

    concept: str
    layers: list[str]
    accuracy: list[float]
    std: list[float]
    n_samples: int
    n_classes: int

    def best_layer(self) -> str:
        """Layer with the highest mean cross-validated accuracy."""
        return self.layers[max(range(len(self.layers)), key=lambda i: self.accuracy[i])]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.layers, self.accuracy))

    def plot(self, ax=None):
        from nndbg.viz.plotting import line

        return line(
            self.layers,
            {self.concept: self.accuracy},
            yerr={self.concept: self.std},
            title=f"Probe accuracy by layer — concept: {self.concept!r}",
            xlabel="layer",
            ylabel="cross-validated accuracy",
            ax=ax,
        )

    def plotly(self):
        from nndbg.viz.plotly_backend import line

        return line(
            self.layers,
            {self.concept: self.accuracy},
            title=f"Probe accuracy by layer — concept: {self.concept!r}",
            xlabel="layer",
            ylabel="accuracy",
        )

    def __repr__(self) -> str:
        best = self.best_layer()
        best_acc = self.accuracy[self.layers.index(best)]
        return (
            f"ProbeResult(concept={self.concept!r}, n_layers={len(self.layers)}, "
            f"best_layer={best!r}, best_accuracy={best_acc:.3f})"
        )
