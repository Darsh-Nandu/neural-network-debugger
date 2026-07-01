"""Result type for the neuron analysis plane."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class NeuronResult:
    """Statistics about individual neurons (hidden-dim features) in a layer.

    Attributes:
        layer: name of the probed layer.
        mean_activation: per-neuron mean activation across the dataset.
        max_activation: per-neuron maximum activation across the dataset.
        std_activation: per-neuron standard deviation.
        kurtosis: excess kurtosis per neuron — high kurtosis = sparse /
            monosemantic; low kurtosis = broad / polysemantic.
        dead_mask: boolean mask of neurons that never fired above threshold.
        top_examples: mapping from neuron index to the list of example
            indices that produced the highest activation for that neuron.
    """

    layer: str
    mean_activation: torch.Tensor    # (n_neurons,)
    max_activation: torch.Tensor     # (n_neurons,)
    std_activation: torch.Tensor     # (n_neurons,)
    kurtosis: torch.Tensor           # (n_neurons,)
    dead_mask: torch.Tensor          # (n_neurons,) bool
    top_examples: dict[int, list[int]] = field(default_factory=dict)

    @property
    def dead_count(self) -> int:
        return int(self.dead_mask.sum())

    @property
    def n_neurons(self) -> int:
        return len(self.dead_mask)

    def polysemantic_neurons(self, percentile: float = 25.0) -> list[int]:
        """Neuron indices whose kurtosis falls below ``percentile`` — these
        neurons activate broadly and are likely polysemantic."""
        threshold = float(torch.quantile(self.kurtosis.float(), percentile / 100.0))
        return (self.kurtosis < threshold).nonzero(as_tuple=True)[0].tolist()

    def plot(self, ax=None):
        """Bar chart of mean activation with dead neurons highlighted."""
        import matplotlib.patches as mpatches

        from nndbg.viz.plotting import _new_ax
        from nndbg.viz.style import apply_style

        fig, ax = _new_ax(ax)
        vals = self.mean_activation.numpy()
        colors = ["#e74c3c" if dead else "#3498db" for dead in self.dead_mask.tolist()]
        ax.bar(range(len(vals)), vals, color=colors, width=1.0)
        ax.set_title(f"Neuron mean activation — {self.layer}  ({self.dead_count} dead / {self.n_neurons})")
        ax.set_xlabel("neuron index")
        ax.set_ylabel("mean activation")
        apply_style(ax)
        ax.legend(
            handles=[
                mpatches.Patch(color="#3498db", label="alive"),
                mpatches.Patch(color="#e74c3c", label="dead"),
            ],
            fontsize=8,
        )
        fig.tight_layout()
        return ax

    def plot_kurtosis(self, ax=None):
        """Bar chart of per-neuron excess kurtosis (polysemanticity proxy)."""
        from nndbg.viz.plotting import bar

        vals = self.kurtosis.numpy()
        return bar(
            [str(i) for i in range(len(vals))],
            vals,
            title=f"Neuron kurtosis — {self.layer}  (high = monosemantic)",
            xlabel="neuron index",
            ylabel="excess kurtosis",
            ax=ax,
        )

    def __repr__(self) -> str:
        return (
            f"NeuronResult(layer={self.layer!r}, n_neurons={self.n_neurons}, "
            f"dead={self.dead_count}/{self.n_neurons}, "
            f"mean_kurtosis={self.kurtosis.mean().item():.2f})"
        )
