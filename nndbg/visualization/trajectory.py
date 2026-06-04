from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nndbg.utils.logging import get_logger
from nndbg.utils.console import console

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


logger = get_logger(__name__)


@dataclass
class LayerTrace:
    """Stats for one layer in a trace."""
    name: str
    mean: float
    std: float
    l2_norm: float
    sparsity: float
    min_val: float
    max_val: float
    tensor: Optional[torch.Tensor] = field(default=None, repr=False)


class TraceResults:
    """
    Returned by ModelProbe.trace().

    Contains layer-by-layer activation stats for a single input sentence.

    Attributes:
        text       : the input sentence
        model_name : model used
        layers     : list of LayerTrace objects in order
        activations: raw dict {layer_name -> tensor}

    Example:
        trace = probe.trace("Le chat était assis.")
        trace.show()
        trace.summary()
        trace.most_active(top_k=5)
        trace.stable_at()

        # Raw tensors for custom analysis
        tensor = trace.activations["encoder.layer.4"]

        # Compare two sentences
        english = probe.trace("The cat sat on the mat.")
        french  = probe.trace("Le chat était assis.")
        english.compare(french)
    """

    def __init__(
        self,
        text: str,
        model_name: str,
        layers: List[LayerTrace],
        activations: Dict[str, torch.Tensor],
    ):
        self.text = text
        self.model_name = model_name
        self.layers = layers
        self.activations = activations  # {layer_name -> tensor}

    def most_active(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Top-k layers by l2_norm — the most strongly activated layers.

        Returns:
            [(layer_name, l2_norm), ...]  sorted highest → lowest
        """
        ranked = sorted(
            self.layers,
            key=lambda l: l.l2_norm,
            reverse=True,
        )
        return [(l.name, l.l2_norm) for l in ranked[:top_k]]

    def stable_at(self, threshold: float = 0.01) -> Optional[str]:
        """
        Find the layer where the representation stabilizes —
        where the change in mean activation drops below threshold.

        Args:
            threshold: minimum change to be considered "still evolving"

        Returns:
            layer name where stabilization occurs, or None
        """
        if len(self.layers) < 2:
            return None

        for i in range(1, len(self.layers)):
            delta = abs(self.layers[i].mean - self.layers[i - 1].mean)
            if delta < threshold:
                return self.layers[i].name

        return self.layers[-1].name

    def summary(self) -> str:
        """Rich-formatted table of every layer's activation stats."""

        header = Text()
        header.append("Model  ", style="dim")
        header.append(f"{self.model_name}\n", style="bold white")
        header.append("Input  ", style="dim")
        header.append(f'"{self.text[:60]}"\n', style="bold yellow")
        header.append("Layers ", style="dim")
        header.append(str(len(self.layers)), style="bold cyan")

        stable = self.stable_at()
        if stable:
            header.append("\nStable ", style="dim")
            header.append(f"at {stable}", style="bold green")

        console.print()
        console.print(Panel(
            header,
            title="[bold cyan]Activation Trajectory[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        ))

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 2),
            expand=True,
        )

        table.add_column("Layer",    style="white",  ratio=3)
        table.add_column("Mean",     style="magenta",width=8,  justify="right")
        table.add_column("Std",      style="blue",   width=8,  justify="right")
        table.add_column("L2 Norm",  style="green",  width=9,  justify="right")
        table.add_column("Sparsity", style="yellow", width=9,  justify="right")
        table.add_column("Activity", ratio=2)

        # Find max l2 for relative bar scaling
        max_l2 = max((l.l2_norm for l in self.layers), default=1.0) or 1.0

        for layer in self.layers:
            # Color by activity level
            ratio = layer.l2_norm / max_l2
            if ratio >= 0.8:
                bar_style = "bold green"
            elif ratio >= 0.5:
                bar_style = "bold yellow"
            else:
                bar_style = "dim white"

            # Activity bar
            filled = int(ratio * 20)
            empty  = 20 - filled
            bar = Text()
            bar.append("█" * filled, style=bar_style)
            bar.append("░" * empty,  style="dim")

            # Highlight stabilization layer
            is_stable = (stable and layer.name == stable)
            row_style = "on dark_green" if is_stable else ""

            table.add_row(
                Text(layer.name, style=f"white {row_style}"),
                Text(f"{layer.mean:.4f}",    style=f"magenta {row_style}"),
                Text(f"{layer.std:.4f}",     style=f"blue {row_style}"),
                Text(f"{layer.l2_norm:.2f}", style=f"green {row_style}"),
                Text(f"{layer.sparsity:.1%}",style=f"yellow {row_style}"),
                bar,
            )

        console.print(table)
        console.print()
        console.print("[bold cyan]  Most active layers:[/bold cyan]")
        for name, l2 in self.most_active(top_k=5):
            short = ".".join(name.split(".")[-2:]) if "." in name else name
            bar_len = int((l2 / max_l2) * 20)
            console.print(
                f"  [dim]{short:<30}[/dim] "
                f"[green]{'█' * bar_len}[/green] "
                f"[bold green]{l2:.2f}[/bold green]"
            )

        console.print()
        return ""

    def show(self) -> None:
        """
        Interactive Plotly line chart.
        X axis = layers, Y axis = activation stats (mean, std, l2_norm).
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print(self.summary())
            return

        layer_names = [l.name for l in self.layers]
        short_names = [
            ".".join(l.split(".")[-2:]) if "." in l else l
            for l in layer_names
        ]
        means    = [l.mean    for l in self.layers]
        stds     = [l.std     for l in self.layers]
        l2_norms = [l.l2_norm for l in self.layers]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=short_names, y=means,
            mode="lines+markers",
            name="mean activation",
            line=dict(color="rgba(99,110,250,0.9)", width=2),
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>Mean:</b> %{y:.4f}<extra></extra>"
            ),
            customdata=layer_names,
        ))

        fig.add_trace(go.Scatter(
            x=short_names, y=stds,
            mode="lines+markers",
            name="std",
            line=dict(color="rgba(239,85,59,0.9)", width=2),
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>Std:</b> %{y:.4f}<extra></extra>"
            ),
            customdata=layer_names,
        ))

        fig.add_trace(go.Scatter(
            x=short_names, y=l2_norms,
            mode="lines+markers",
            name="l2 norm",
            line=dict(color="rgba(0,204,150,0.9)", width=2),
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>L2 norm:</b> %{y:.4f}<extra></extra>"
            ),
            customdata=layer_names,
        ))

        # Mark stabilization point
        stable = self.stable_at()
        if stable and stable in layer_names:
            idx = layer_names.index(stable)
            fig.add_vline(
                x=idx,
                line_dash="dash",
                line_color="rgba(255,255,255,0.4)",
                annotation_text="stabilizes here",
                annotation_position="top right",
            )

        fig.update_layout(
            title=(
                f"Activation Trajectory<br>"
                f"<sub>Model: {self.model_name} | "
                f"Input: \"{self.text[:50]}\"</sub>"
            ),
            xaxis_title="Layer",
            yaxis_title="Activation value",
            template="plotly_dark",
            font=dict(family="monospace"),
            height=500,
            hovermode="x unified",
        )

        fig.show()

    def compare(self, other: "TraceResults") -> None:
        """
        Overlay this trace against another on the same chart.
        Useful for comparing how two sentences evolve differently.

        Example:
            english = probe.trace("The cat sat on the mat.")
            french  = probe.trace("Le chat était assis.")
            english.compare(french)
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly required for compare(). pip install plotly")
            return

        def _short(names):
            return [
                ".".join(n.split(".")[-2:]) if "." in n else n
                for n in names
            ]

        fig = go.Figure()

        # Trace A (self) 
        names_a  = [l.name for l in self.layers]
        short_a  = _short(names_a)
        means_a  = [l.mean    for l in self.layers]
        l2_a     = [l.l2_norm for l in self.layers]

        fig.add_trace(go.Scatter(
            x=short_a, y=means_a,
            mode="lines+markers",
            name=f'mean — "{self.text[:30]}"',
            line=dict(color="rgba(99,110,250,0.9)", width=2),
            customdata=names_a,
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>Mean:</b> %{y:.4f}<extra></extra>"
            ),
        ))

        fig.add_trace(go.Scatter(
            x=short_a, y=l2_a,
            mode="lines+markers",
            name=f'l2 — "{self.text[:30]}"',
            line=dict(color="rgba(99,110,250,0.4)", width=2, dash="dot"),
            customdata=names_a,
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>L2:</b> %{y:.4f}<extra></extra>"
            ),
        ))

        # ── Trace B (other) ──
        names_b  = [l.name for l in other.layers]
        short_b  = _short(names_b)
        means_b  = [l.mean    for l in other.layers]
        l2_b     = [l.l2_norm for l in other.layers]

        fig.add_trace(go.Scatter(
            x=short_b, y=means_b,
            mode="lines+markers",
            name=f'mean — "{other.text[:30]}"',
            line=dict(color="rgba(239,85,59,0.9)", width=2),
            customdata=names_b,
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>Mean:</b> %{y:.4f}<extra></extra>"
            ),
        ))

        fig.add_trace(go.Scatter(
            x=short_b, y=l2_b,
            mode="lines+markers",
            name=f'l2 — "{other.text[:30]}"',
            line=dict(color="rgba(239,85,59,0.4)", width=2, dash="dot"),
            customdata=names_b,
            hovertemplate=(
                "<b>Layer:</b> %{customdata}<br>"
                "<b>L2:</b> %{y:.4f}<extra></extra>"
            ),
        ))

        # Divergence marker
        # Find the layer where the two means diverge the most
        common = [
            l for l in names_a if l in set(names_b)
        ]
        if common:
            a_map = {l.name: l.mean for l in self.layers}
            b_map = {l.name: l.mean for l in other.layers}
            diffs = {l: abs(a_map[l] - b_map[l]) for l in common}
            max_diverge = max(diffs, key=diffs.get)
            idx = names_a.index(max_diverge)
            fig.add_vline(
                x=idx,
                line_dash="dash",
                line_color="rgba(255,255,255,0.3)",
                annotation_text="max divergence",
                annotation_position="top right",
            )

        fig.update_layout(
            title=(
                f"Activation Comparison<br>"
                f"<sub>"
                f'"{self.text[:40]}" vs "{other.text[:40]}"'
                f"</sub>"
            ),
            xaxis_title="Layer",
            yaxis_title="Activation value",
            template="plotly_dark",
            font=dict(family="monospace"),
            height=550,
            hovermode="x unified",
        )

        fig.show()

    def __repr__(self) -> str:
        return (
            f"TraceResults("
            f"text='{self.text[:40]}', "
            f"layers={len(self.layers)}, "
            f"model='{self.model_name}')"
        )