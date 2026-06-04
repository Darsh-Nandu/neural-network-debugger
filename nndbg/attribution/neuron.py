"""
NeuronAttributor — finds which neurons fire most per concept group.

How it works:
    For each layer, for each group — compute the mean activation per
    neuron across all samples. Then rank neurons by how differently
    they fire between groups (differential activation).

    High differential = neuron is specific to that concept.
    Low differential  = neuron fires equally for everything (not useful).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nndbg.utils.console import console
from nndbg.utils.logging import get_logger

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


logger = get_logger(__name__)


class NeuronAttributor:

    def __init__(self, probe):
        
        self._probe = probe
        self._hook_engine = probe._hook_engine

        # { axis_name -> { group_name -> { layer_name -> mean_per_neuron } } }
        self._neuron_means: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self._fitted_axes: List[str] = []


    def fit(self, axis_name: str) -> "NeuronAttributor":

        # Find the axis
        axis = next(
            (a for a in self._probe._axes if a.name == axis_name), None
        )
        if axis is None:
            raise ValueError(
                f"Axis '{axis_name}' not found. "
                f"Available: {[a.name for a in self._probe._axes]}"
            )

        logger.info(f"NeuronAttributor fitting axis: '{axis_name}'")

        self._neuron_means[axis_name] = {}
        self._hook_engine.attach()

        for group_name, texts in axis.groups.items():
            logger.info(
                f"  Collecting neurons for group '{group_name}'"
            )
            self._neuron_means[axis_name][group_name] = {}

            # layer_name -> list of per-neuron arrays (one per sample)
            layer_samples: Dict[str, List[np.ndarray]] = {}

            for text in texts:
                activations = self._probe._run_single(text)

                for layer_name, tensor in activations.items():
                    arr = tensor.float().numpy()

                    if arr.ndim > 1:
                        # mean over all dimensions except the last
                        for _ in range(arr.ndim - 1):
                            arr = arr.mean(axis=0)
                    arr = arr.flatten()
                    arr = np.nan_to_num(arr)

                    if layer_name not in layer_samples:
                        layer_samples[layer_name] = []
                    layer_samples[layer_name].append(arr)

            # Average across all samples for this group
            for layer_name, samples in layer_samples.items():
                # Stack and mean → shape (n_neurons,)
                stacked = np.stack(samples, axis=0)
                self._neuron_means[axis_name][group_name][layer_name] = (
                    stacked.mean(axis=0)
                )

        self._hook_engine.detach()
        self._fitted_axes.append(axis_name)
        logger.info(
            f"NeuronAttributor fit complete for axis '{axis_name}'"
        )

        return self


    def top_neurons(
        self,
        axis_name: str,
        group: str,
        layer_name: Optional[str] = None,
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[int, float]]]:

        self._check_fitted(axis_name)
        self._check_group(axis_name, group)

        results = {}
        layers = self._get_layers(axis_name, layer_name)

        for lname in layers:
            target = self._neuron_means[axis_name][group].get(lname)
            if target is None:
                continue

            # Mean of all OTHER groups for this layer
            others = [
                self._neuron_means[axis_name][g][lname]
                for g in self._neuron_means[axis_name]
                if g != group and lname in self._neuron_means[axis_name][g]
            ]

            if not others:
                continue

            other_mean = np.stack(others).mean(axis=0)

            # Differential: how much more does this group fire vs others
            diff = target - other_mean

            # Top-k by absolute differential
            top_indices = np.argsort(diff)[::-1][:top_k]
            results[lname] = [
                (int(idx), float(diff[idx]))
                for idx in top_indices
            ]

        return results


    def exclusive_neurons(
        self,
        axis_name: str,
        group: str,
        layer_name: Optional[str] = None,
        top_k: int = 10,
        min_diff: float = 0.1,
    ) -> Dict[str, List[Tuple[int, float]]]:
        
        self._check_fitted(axis_name)
        self._check_group(axis_name, group)

        results = {}
        layers = self._get_layers(axis_name, layer_name)

        for lname in layers:
            target = self._neuron_means[axis_name][group].get(lname)
            if target is None:
                continue

            others = [
                self._neuron_means[axis_name][g][lname]
                for g in self._neuron_means[axis_name]
                if g != group and lname in self._neuron_means[axis_name][g]
            ]

            if not others:
                continue

            other_mean  = np.stack(others).mean(axis=0)
            other_max   = np.stack(others).max(axis=0)

            # Exclusivity: fires high for target, low for ALL others
            exclusivity = target - other_max
            exclusivity = np.where(exclusivity > min_diff, exclusivity, 0)

            top_indices = np.argsort(exclusivity)[::-1][:top_k]
            top_indices = [
                idx for idx in top_indices if exclusivity[idx] > 0
            ]

            if top_indices:
                results[lname] = [
                    (int(idx), float(exclusivity[idx]))
                    for idx in top_indices
                ]

        return results


    def show(
        self,
        axis_name: str,
        group: str,
        layer_name: Optional[str] = None,
        top_k: int = 10,
        mode: str = "top",
    ) -> None:

        if mode == "exclusive":
            data = self.exclusive_neurons(
                axis_name, group, layer_name, top_k
            )
            title = f"Exclusive Neurons — '{group}' in axis '{axis_name}'"
        else:
            data = self.top_neurons(
                axis_name, group, layer_name, top_k
            )
            title = f"Top Neurons — '{group}' in axis '{axis_name}'"

        if not data:
            console.print("[yellow]No neurons found.[/yellow]")
            return

        console.print()
        console.print(Panel(
            f"[dim]Group:[/dim] [bold blue]{group}[/bold blue]   "
            f"[dim]Axis:[/dim] [bold cyan]{axis_name}[/bold cyan]   "
            f"[dim]Mode:[/dim] [white]{mode}[/white]",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 4),
        ))

        for lname, neurons in data.items():
            if not neurons:
                continue

            console.print(
                f"\n  [bold white]{lname}[/bold white]"
            )

            table = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
            )
            table.add_column("Neuron",     width=10,  justify="right")
            table.add_column("Differential", width=12, justify="right")
            table.add_column("Activity",   ratio=1)

            max_diff = max(abs(score) for _, score in neurons) or 1.0

            for neuron_idx, score in neurons:
                bar_len = int((abs(score) / max_diff) * 24)

                if score >= 0:
                    bar_style  = "bold green"
                    score_style = "bold green"
                else:
                    bar_style  = "bold red"
                    score_style = "bold red"

                bar = Text()
                bar.append("█" * bar_len, style=bar_style)
                bar.append("░" * (24 - bar_len), style="dim")

                table.add_row(
                    Text(f"#{neuron_idx}", style="white"),
                    Text(f"{score:+.4f}", style=score_style),
                    bar,
                )

            console.print(table)

        console.print()


    def _check_fitted(self, axis_name: str) -> None:
        if axis_name not in self._fitted_axes:
            raise ValueError(
                f"Axis '{axis_name}' not fitted yet. "
                f"Call attr.fit('{axis_name}') first."
            )


    def _check_group(self, axis_name: str, group: str) -> None:
        available = list(self._neuron_means[axis_name].keys())
        if group not in available:
            raise ValueError(
                f"Group '{group}' not found in axis '{axis_name}'. "
                f"Available: {available}"
            )


    def _get_layers(
        self,
        axis_name: str,
        layer_name: Optional[str],
    ) -> List[str]:
        """Return layers to process — one specific or all."""
        all_layers = list(
            next(iter(self._neuron_means[axis_name].values())).keys()
        )
        if layer_name is not None:
            if layer_name not in all_layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {all_layers[:5]}..."
                )
            return [layer_name]
        return all_layers


    def __repr__(self) -> str:
        return (
            f"NeuronAttributor("
            f"fitted_axes={self._fitted_axes})"
        )