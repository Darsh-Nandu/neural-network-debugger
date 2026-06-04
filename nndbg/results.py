"""
ProbeResults — holds and visualizes output from a ModelProbe run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from nndbg.probing.axis import Axis
from nndbg.storage.store import ActivationStore
from nndbg.utils.logging import get_logger
from nndbg.utils.console import console

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box


logger = get_logger(__name__)


class ProbeResults:
    """
    Returned by ModelProbe.run().

    Contains:
        - probe_scores  : which layers encode which concepts
        - activation stats per layer per group
        - visualization methods
    """

    def __init__(
        self,
        run_id: str,
        model_name: str,
        axes: List[Axis],
        probe_scores: Dict[str, Dict[str, float]],
        store: ActivationStore,
        layer_group_data: Dict,
    ):
        self.run_id = run_id
        self.model_name = model_name
        self.axes = axes
        self.probe_scores = probe_scores
        self._store = store
        self._layer_group_data = layer_group_data

    def encoding_layers(
        self,
        axis_name: str,
        top_k: Optional[int] = 5,
        min_score: float = 0.0,
    ) -> List[tuple]:
        """
        Return layers ranked by how well they encode an axis.

        Args:
            axis_name:  which axis to query e.g. "language"
            top_k:      how many layers to return.
                        Pass None to get ALL layers.
            min_score:  only return layers above this accuracy threshold.
                        e.g. min_score=0.7 returns only strong encoders.

        Returns:
            [(layer_name, probe_accuracy), ...]  sorted best → worst

        Examples:
            # Top 5 layers (default)
            results.encoding_layers("language")

            # All layers
            results.encoding_layers("language", top_k=None)

            # All layers above 70% accuracy
            results.encoding_layers("language", top_k=None, min_score=0.7)

            # Top 10 layers
            results.encoding_layers("language", top_k=10)
        """
        if axis_name not in self.probe_scores:
            raise ValueError(
                f"Axis '{axis_name}' not found. "
                f"Available: {list(self.probe_scores.keys())}"
            )

        scores = self.probe_scores[axis_name]

        # Filter by min_score
        filtered = {
            layer: score
            for layer, score in scores.items()
            if score >= min_score
        }

        # Sort best first
        ranked = sorted(
            filtered.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply top_k (None means return all)
        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def summary(
        self,
        top_k: Optional[int] = 5,
        min_score: float = 0.0,
    ) -> str:
        """
        Rich-formatted report of findings.

        Args:
            top_k:      layers to show per axis. None = all layers.
            min_score:  only show layers above this threshold.
        """
        header = Text()
        header.append("Model  ", style="dim")
        header.append(f"{self.model_name}\n", style="bold white")
        header.append("Run    ", style="dim")
        header.append(f"{self.run_id}", style="bold cyan")

        console.print()
        console.print(Panel(
            header,
            title="[bold cyan]NNDbg Analysis Report[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        ))

        for axis in self.axes:
            layers = self.encoding_layers(
                axis.name,
                top_k=top_k,
                min_score=min_score,
            )

            # Axis info panel
            info = Text()
            info.append("Groups   ", style="dim")
            info.append(f"{', '.join(axis.group_names)}\n", style="bold blue")
            info.append("Samples  ", style="dim")
            info.append(f"{axis.total_samples}\n", style="white")
            info.append("Showing  ", style="dim")
            info.append(
                f"{'all' if top_k is None else f'top {top_k}'} layers",
                style="white",
            )
            if min_score > 0:
                info.append(f"  (min score ≥ {min_score})", style="dim")

            console.print()
            console.print(Panel(
                info,
                title=f"[bold blue]Axis: {axis.name}[/bold blue]",
                border_style="blue",
                padding=(0, 4),
            ))

            if not layers:
                console.print(
                    f"  [yellow]No layers found above min_score={min_score}[/yellow]"
                )
                continue

            # Results table
            table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
                expand=True,
            )

            table.add_column("#",          style="dim",         width=4,  justify="right")
            table.add_column("Layer",      style="white",       ratio=3)
            table.add_column("Score",      style="bold magenta",width=8,  justify="right")
            table.add_column("Confidence", ratio=2)

            for rank, (layer, score) in enumerate(layers, 1):
                # Color the score
                if score >= 0.8:
                    score_style = "bold green"
                elif score >= 0.6:
                    score_style = "bold yellow"
                else:
                    score_style = "bold red"

                # Build progress bar
                filled  = int(score * 24)
                empty   = 24 - filled
                bar = Text()
                bar.append("█" * filled, style=score_style)
                bar.append("░" * empty,  style="dim")
                bar.append(f"  {score:.1%}", style=score_style)

                table.add_row(
                    str(rank),
                    layer,
                    Text(f"{score:.3f}", style=score_style),
                    bar,
                )

            console.print(table)

        console.print()
        return "" 
    
    def show(
        self,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> None:
        """
        Interactive bar charts — one per axis.

        Args:
            top_k:      how many layers to show. None = all.
            min_score:  only show layers above this threshold.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print(self.summary(top_k=top_k, min_score=min_score))
            return

        n = len(self.axes)
        fig = make_subplots(
            rows=n,
            cols=1,
            subplot_titles=[f"Axis: {ax.name}" for ax in self.axes],
            vertical_spacing=0.12,
        )

        for row, axis in enumerate(self.axes, 1):
            ranked = self.encoding_layers(
                axis.name,
                top_k=top_k,
                min_score=min_score,
            )
            if not ranked:
                continue

            layers, values = zip(*ranked)
            short = [
                ".".join(l.split(".")[-2:]) if "." in l else l
                for l in layers
            ]

            fig.add_trace(
                go.Bar(
                    x=short,
                    y=values,
                    name=axis.name,
                    marker_color=[
                        f"rgba(99,110,250,{0.3 + 0.7 * v})"
                        for v in values
                    ],
                    hovertemplate=(
                        "<b>Layer:</b> %{customdata}<br>"
                        "<b>Probe accuracy:</b> %{y:.3f}"
                        "<extra></extra>"
                    ),
                    customdata=layers,
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(
                title_text="Probe accuracy",
                range=[0, 1],
                row=row,
                col=1,
            )

        fig.update_layout(
            title=(
                f"NNDbg — Semantic Encoding Analysis"
                f"<br><sub>Model: {self.model_name}</sub>"
            ),
            height=420 * n,
            template="plotly_dark",
            font=dict(family="monospace"),
            showlegend=True,
        )
        fig.show()

    def plot_heatmap(
        self,
        min_score: float = 0.0,
    ) -> None:
        """
        Heatmap: layers × axes, colored by probe accuracy.

        Args:
            min_score: only show layers where at least one axis
                       scores above this threshold.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print(self.summary())
            return

        all_layers: set = set()
        for axis in self.axes:
            for layer, score in self.probe_scores.get(axis.name, {}).items():
                if score >= min_score:
                    all_layers.add(layer)

        if not all_layers:
            print(f"No layers found above min_score={min_score}")
            return

        layers = sorted(all_layers)
        axis_names = [ax.name for ax in self.axes]

        matrix = np.zeros((len(layers), len(axis_names)))
        l_idx = {l: i for i, l in enumerate(layers)}

        for j, axis in enumerate(self.axes):
            for layer, score in self.probe_scores.get(axis.name, {}).items():
                if layer in l_idx:
                    matrix[l_idx[layer], j] = score

        short_layers = [
            ".".join(l.split(".")[-2:]) if "." in l else l
            for l in layers
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=axis_names,
                y=short_layers,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Probe accuracy"),
                hovertemplate=(
                    "<b>Layer:</b> %{customdata}<br>"
                    "<b>Axis:</b> %{x}<br>"
                    "<b>Score:</b> %{z:.3f}"
                    "<extra></extra>"
                ),
                customdata=[[l] * len(axis_names) for l in layers],
            )
        )

        fig.update_layout(
            title=f"Layer encoding map — {self.model_name}",
            xaxis_title="Concept axis",
            yaxis_title="Layer",
            height=max(500, 22 * len(layers)),
            template="plotly_dark",
            font=dict(family="monospace"),
        )
        fig.show()

    def to_dict(self) -> dict:
        return {
            "run_id":       self.run_id,
            "model_name":   self.model_name,
            "axes":         [ax.name for ax in self.axes],
            "probe_scores": self.probe_scores,
        }

    def __repr__(self) -> str:
        axes = [ax.name for ax in self.axes]
        return (
            f"ProbeResults("
            f"run_id='{self.run_id}', "
            f"model='{self.model_name}', "
            f"axes={axes})"
        )