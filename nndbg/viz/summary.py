"""Rich-based terminal summary for Inspector.summary()."""
from __future__ import annotations

from typing import TYPE_CHECKING

from nndbg.utils.console import console

if TYPE_CHECKING:
    from nndbg.inspector import Inspector

_PLANES = [
    ("probing", "Where is concept X encoded? - linear probes per layer"),
    ("attribution", "Which inputs caused this output? - saliency / IG / GradCAM"),
    ("attention", "What does each head attend to? - per-head heatmaps, rollout"),
    ("patching", "Which layers causally produce a behaviour? - activation patching"),
    ("sae", "What sparse features does this layer learn? - sparse autoencoder"),
    ("latent", "Where do activations sit in a compressed latent space? - VAE"),
]


def print_summary(inspector: Inspector) -> None:
    from rich.table import Table

    model_name = type(inspector.model).__name__
    n_params = sum(p.numel() for p in inspector.model.parameters())
    n_layers = len(inspector.layers())

    console.print(
        f"[model]{model_name}[/model]  "
        f"[layer]{n_layers} layers[/layer]  "
        f"[score]{n_params:,} parameters[/score]  "
        f"[dim]device={inspector.device}[/dim]"
    )

    table = Table(title="Analysis planes", show_lines=False)
    table.add_column("inspector.<plane>", style="header")
    table.add_column("answers")
    for name, description in _PLANES:
        table.add_row(name, description)
    console.print(table)
