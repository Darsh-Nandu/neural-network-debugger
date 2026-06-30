"""Shared activation-collection helper used by probing, SAE, and VAE planes."""
from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn

from nndbg.core.hooks import HookManager

Pooling = Literal["last", "mean", "first"]


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    hooks: HookManager,
    inputs: Iterable[torch.Tensor],
    layers: list[str],
    *,
    pooling: Pooling = "last",
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Run ``model`` on each item in ``inputs``, capture activations at
    ``layers``, pool each sequence to a single vector, and stack into
    ``{layer: (n_inputs, hidden_dim)}``.

    Args:
        inputs: an iterable of token-id tensors, one per example (shape
            ``(seq,)`` or ``(1, seq)``).
        pooling: how to collapse the sequence dimension — last token,
            mean over positions, or first token.
    """
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in layers}

    for item in inputs:
        if device is not None:
            item = item.to(device)
        if item.dim() == 1:
            item = item.unsqueeze(0)

        with hooks.capture(layers) as cache:
            model(item)

        for name in layers:
            pooled = _pool(cache[name], pooling)
            collected[name].append(pooled.squeeze(0))

    return {name: torch.stack(vecs) for name, vecs in collected.items()}


def _pool(tensor: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    if tensor.dim() == 2:  # already (batch, hidden) — e.g. a pooler output
        return tensor
    if pooling == "last":
        return tensor[:, -1, :]
    if pooling == "mean":
        return tensor.mean(dim=1)
    if pooling == "first":
        return tensor[:, 0, :]
    raise ValueError(f"Unknown pooling strategy: {pooling!r}")
