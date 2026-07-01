"""NeuronAnalyzer — what is each neuron doing?

Collects per-neuron activation statistics across a dataset to surface
dead neurons (never activate), polysemantic neurons (activate broadly
rather than sharply), and which examples most strongly drive each neuron.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from nndbg.analysis.neurons.results import NeuronResult
from nndbg.core.collect import collect_activations

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


def _kurtosis(X: torch.Tensor) -> torch.Tensor:
    """Per-column excess kurtosis of (n_samples, n_neurons) tensor."""
    X = X.float()
    mu = X.mean(0)
    std = X.std(0).clamp_min(1e-8)
    z = (X - mu) / std
    return (z ** 4).mean(0) - 3.0


class NeuronAnalyzer:
    """Per-neuron statistics for a single layer.

    Example::

        result = inspector.neurons.stats(dataset, layer="transformer.h.5.mlp.c_fc")
        print(result)           # dead count, mean kurtosis
        result.plot()           # bar chart: mean activation, dead neurons in red
        result.plot_kurtosis()  # polysemanticity proxy per neuron

        # Which examples most activated neuron 42?
        result = inspector.neurons.top_activating(dataset, layer=..., neuron=42, k=5)
        print(result.top_examples[42])
    """

    def __init__(self, inspector: Inspector) -> None:
        self._inspector = inspector

    def stats(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        pooling: str = "last",
        dead_threshold: float = 1e-2,
        top_k: int = 10,
    ) -> NeuronResult:
        """Compute per-neuron statistics across the full dataset.

        Args:
            layer: the layer to analyse.
            pooling: how to reduce the sequence dim for 3-D activations.
            dead_threshold: a neuron is considered dead if its maximum
                activation across all examples is below this value.
            top_k: how many top-activating example indices to record per
                neuron.
        """
        inspector = self._inspector
        acts = collect_activations(
            inspector.model, inspector._hooks, dataset, [layer], pooling=pooling, device=inspector.device
        )
        X = acts[layer].float().cpu()  # (n_examples, n_neurons)

        mean_act = X.mean(0)
        max_act = X.max(0).values
        std_act = X.std(0)
        kurt = _kurtosis(X)
        dead_mask = max_act < dead_threshold

        k = min(top_k, X.shape[0])
        top_idx = X.topk(k, dim=0).indices  # (k, n_neurons)
        top_examples = {
            neuron_i: top_idx[:, neuron_i].tolist()
            for neuron_i in range(X.shape[1])
        }

        return NeuronResult(
            layer=layer,
            mean_activation=mean_act,
            max_activation=max_act,
            std_activation=std_act,
            kurtosis=kurt,
            dead_mask=dead_mask,
            top_examples=top_examples,
        )

    def top_activating(
        self,
        dataset: Sequence[torch.Tensor],
        *,
        layer: str,
        pooling: str = "last",
        k: int = 10,
        neuron: int | None = None,
    ) -> NeuronResult:
        """Convenience alias: run ``stats`` and return only the ``top_k``
        examples per neuron (or for a single ``neuron`` if specified).

        Equivalent to ``stats(..., top_k=k)`` but makes intent explicit when
        you only care about top-activating examples.
        """
        result = self.stats(dataset, layer=layer, pooling=pooling, top_k=k)
        if neuron is not None:
            if neuron not in result.top_examples:
                raise ValueError(f"neuron={neuron} out of range for layer {layer!r} (0–{result.n_neurons - 1})")
            result.top_examples = {neuron: result.top_examples[neuron]}
        return result
