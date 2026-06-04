"""
ProbeResults — holds and visualizes output from a ModelProbe run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from nndbg.probing.axis import Axis
from nndbg.storage.store import ActivationStore
from nndbg.utils.logging import get_logger

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

    # ------------------------------------------------------------------
    # Key Insights
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Text Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        top_k: Optional[int] = 5,
        min_score: float = 0.0,
    ) -> str:
        """
        Human-readable report of findings.

        Args:
            top_k:      layers to show per axis. None = all layers.
            min_score:  only show layers above this threshold.
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  NNDbg Analysis Report")
        lines.append(f"  Model : {self.model_name}")
        lines.append(f"  Run   : {self.run_id}")
        lines.append(f"{'='*60}")

        for axis in self.axes:
            layers = self.encoding_layers(
                axis.name,
                top_k=top_k,
                min_score=min_score,
            )

            lines.append(f"\n  Axis    : '{axis.name}'")
            lines.append(f"  Groups  : {', '.join(axis.group_names)}")
            lines.append(f"  Samples : {axis.total_samples}")
            lines.append(
                f"  Showing : "
                f"{'all' if top_k is None else f'top {top_k}'} layers"
                + (f" (min_score ≥ {min_score})" if min_score > 0 else "")
            )
            lines.append(f"\n  Encoding layers:")

            if not layers:
                lines.append(
                    f"    No layers found above min_score={min_score}"
                )
            else:
                for rank, (layer, score) in enumerate(layers, 1):
                    bar = "█" * int(score * 20)
                    lines.append(
                        f"    {rank:>3}. {layer:<45} {score:.3f}  {bar}"
                    )

        lines.append(f"\n{'='*60}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

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