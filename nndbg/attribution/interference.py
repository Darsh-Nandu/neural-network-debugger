"""
InterferenceDetector — finds neurons shared across concept axes.

If axis A and axis B share the same neurons, fine-tuning on one
will likely degrade performance on the other.

How it works:
    Take top-k neurons for axis A and top-k neurons for axis B.
    Measure the overlap. High overlap = high interference risk.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from nndbg.utils.console import console
from nndbg.utils.logging import get_logger

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

logger = get_logger(__name__)

# Risk thresholds
_RISK_LEVELS = [
    (0.6, "HIGH",   "bold red",    "Fine-tuning on one WILL hurt the other."),
    (0.3, "MEDIUM", "bold yellow", "Fine-tuning on one MAY affect the other."),
    (0.0, "LOW",    "bold green",  "Concepts are largely independent."),
]


class InterferenceDetector:
    
    def __init__(self, attributor: "NeuronAttributor"):
        """
        Args:
            attributor: a fitted NeuronAttributor instance
        """
        self._attributor = attributor


    def check(
        self,
        axis_a: str,
        axis_b: str,
        top_k: int = 50,
        layer_name: Optional[str] = None,
    ) -> Dict:

        self._attributor._check_fitted(axis_a)
        self._attributor._check_fitted(axis_b)

        layers_a = self._attributor._get_layers(axis_a, layer_name)
        layers_b = self._attributor._get_layers(axis_b, layer_name)
        common_layers = [l for l in layers_a if l in set(layers_b)]

        if not common_layers:
            logger.warning("No common layers found between axes.")
            return {}

        total_shared    = 0
        total_possible  = 0
        shared_layers   = []

        for lname in common_layers:
            # Get top neurons for each axis across all groups
            neurons_a = self._get_top_neurons_for_layer(
                axis_a, lname, top_k
            )
            neurons_b = self._get_top_neurons_for_layer(
                axis_b, lname, top_k
            )

            if not neurons_a or not neurons_b:
                continue

            shared = neurons_a & neurons_b
            n_shared = len(shared)

            if n_shared > 0:
                shared_layers.append((lname, n_shared, shared))

            total_shared   += n_shared
            total_possible += max(len(neurons_a), len(neurons_b))

        overlap_score = (
            total_shared / total_possible
            if total_possible > 0 else 0.0
        )

        risk_level, risk_style, risk_msg = self._get_risk(overlap_score)

        result = {
            "axis_a":        axis_a,
            "axis_b":        axis_b,
            "overlap_score": overlap_score,
            "total_shared":  total_shared,
            "total_neurons": total_possible,
            "risk_level":    risk_level,
            "risk_message":  risk_msg,
            "shared_layers": shared_layers,
        }

        self._print_report(result, risk_style)
        return result


    def check_all(self, top_k: int = 50) -> List[Dict]:
        """
        Check interference between every pair of fitted axes.

        Returns:
            list of result dicts sorted by overlap_score descending
        """
        axes = self._attributor._fitted_axes
        if len(axes) < 2:
            console.print(
                "[yellow]Need at least 2 fitted axes to check interference.[/yellow]"
            )
            return []

        results = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                result = self.check(axes[i], axes[j], top_k=top_k)
                if result:
                    results.append(result)

        results.sort(key=lambda x: x["overlap_score"], reverse=True)
        return results


    def _print_report(self, result: Dict, risk_style: str) -> None:
        """Rich-formatted interference report."""

        overlap = result["overlap_score"]
        bar_len = int(overlap * 30)

        bar = Text()
        bar.append("█" * bar_len,        style=risk_style)
        bar.append("░" * (30 - bar_len),  style="dim")

        header = Text()
        header.append("Axes      ", style="dim")
        header.append(
            f"{result['axis_a']} × {result['axis_b']}\n",
            style="bold white",
        )
        header.append("Shared    ", style="dim")
        header.append(
            f"{result['total_shared']} / {result['total_neurons']} neurons\n",
            style="bold white",
        )
        header.append("Overlap   ", style="dim")
        header.append(f"{overlap:.1%}  ", style=risk_style)
        header.append_text(bar)
        header.append(f"\nRisk      ", style="dim")
        header.append(result["risk_level"], style=risk_style)
        header.append(f"\n          ", style="dim")
        header.append(result["risk_message"], style="dim")

        console.print()
        console.print(Panel(
            header,
            title="[bold cyan]Concept Interference Report[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        ))

        if result["shared_layers"]:
            console.print(
                "\n  [bold white]Layers with shared neurons:[/bold white]"
            )

            table = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
            )
            table.add_column("Layer",         ratio=3, style="white")
            table.add_column("Shared Neurons",width=16, justify="right")

            for lname, n_shared, _ in sorted(
                result["shared_layers"],
                key=lambda x: x[1],
                reverse=True,
            )[:10]:
                short = (
                    ".".join(lname.split(".")[-2:])
                    if "." in lname else lname
                )
                table.add_row(
                    short,
                    Text(str(n_shared), style=risk_style),
                )

            console.print(table)

        console.print()


    def _get_top_neurons_for_layer(
        self,
        axis_name: str,
        layer_name: str,
        top_k: int,
    ) -> set:
        """Get top-k neuron indices for an axis at a specific layer."""
        group_means = self._attributor._neuron_means.get(
            axis_name, {}
        )

        all_diffs = []
        groups = list(group_means.keys())

        for i, g in enumerate(groups):
            target = group_means[g].get(layer_name)
            if target is None:
                continue
            others = [
                group_means[g2][layer_name]
                for g2 in groups
                if g2 != g and layer_name in group_means[g2]
            ]
            if not others:
                continue
            other_mean = np.stack(others).mean(axis=0)
            diff = np.abs(target - other_mean)
            all_diffs.append(diff)

        if not all_diffs:
            return set()

        combined = np.stack(all_diffs).mean(axis=0)
        top_indices = np.argsort(combined)[::-1][:top_k]
        return set(top_indices.tolist())


    def _get_risk(
        self,
        score: float,
    ) -> Tuple[str, str, str]:
        for threshold, level, style, msg in _RISK_LEVELS:
            if score >= threshold:
                return level, style, msg
        return "LOW", "bold green", "Concepts are largely independent."


    def __repr__(self) -> str:
        return (
            f"InterferenceDetector("
            f"axes={self._attributor._fitted_axes})"
        )