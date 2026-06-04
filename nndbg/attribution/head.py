"""
HeadAttributor — finds which attention heads specialize per concept.

How it works:
    Hooks specifically into attention weight outputs.
    For each head in each layer — measures how much the attention
    pattern changes between groups using Jensen-Shannon divergence.

    High score = head pays very different attention for different groups.
    Low score  = head behaves the same regardless of input group.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nndbg.utils.console import console
from nndbg.utils.logging import get_logger

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


logger = get_logger(__name__)


class HeadAttributor:
    """
    Finds attention heads that specialize per concept axis.

    Usage:
        heads = HeadAttributor(probe)
        heads.fit("language")

        # Top specialized heads
        heads.top_heads("language", top_k=5)

        # Full heatmap
        heads.show("language")
    """

    def __init__(self, probe):
        self._probe = probe
        self._hook_engine = probe._hook_engine

        # { axis_name -> { layer_name -> { head_idx -> { group -> mean_attn } } } }
        self._head_means: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {}
        self._fitted_axes: List[str] = []


    def fit(self, axis_name: str) -> "HeadAttributor":

        axis = next(
            (a for a in self._probe._axes if a.name == axis_name), None
        )
        if axis is None:
            raise ValueError(
                f"Axis '{axis_name}' not found. "
                f"Available: {[a.name for a in self._probe._axes]}"
            )

        logger.info(f"HeadAttributor fitting axis: '{axis_name}'")

        self._head_means[axis_name] = {}

        # Only hook attention layers
        def is_attention(name: str, module: nn.Module) -> bool:
            return "attention" in name.lower()

        self._hook_engine.attach(layer_filter=is_attention)

        # First pass: collect all activations and find global max shapes per layer
        all_group_activations: Dict[str, Dict[str, List[np.ndarray]]] = {}
        global_max_shapes: Dict[str, Tuple] = {}

        for group_name, texts in axis.groups.items():
            logger.info(
                f"  Collecting attention for group '{group_name}'"
            )

            # layer -> list of attention tensors
            layer_attns: Dict[str, List[np.ndarray]] = {}

            for text in texts:
                activations = self._probe._run_single(text)

                for layer_name, tensor in activations.items():
                    arr = tensor.float().numpy()
                    arr = np.nan_to_num(arr)

                    if layer_name not in layer_attns:
                        layer_attns[layer_name] = []
                    layer_attns[layer_name].append(arr)
                    
                    # Track global max shape per layer
                    if layer_name not in global_max_shapes:
                        global_max_shapes[layer_name] = arr.shape
                    else:
                        old_shape = global_max_shapes[layer_name]
                        new_shape = tuple(max(old_shape[i], arr.shape[i]) for i in range(len(arr.shape)))
                        global_max_shapes[layer_name] = new_shape
            
            all_group_activations[group_name] = layer_attns

        # Second pass: pad all tensors to global max shapes and compute per-head means
        for group_name, layer_attns in all_group_activations.items():
            for layer_name, tensors in layer_attns.items():
                if layer_name not in self._head_means[axis_name]:
                    self._head_means[axis_name][layer_name] = {}

                # Pad all tensors to the global maximum shape for this layer
                max_shape = global_max_shapes[layer_name]
                padded_tensors = []
                
                for tensor in tensors:
                    if tensor.shape != max_shape:
                        # Pad with zeros
                        pad_widths = [(0, max_shape[i] - tensor.shape[i]) for i in range(len(tensor.shape))]
                        tensor = np.pad(tensor, pad_widths, mode='constant', constant_values=0)
                    padded_tensors.append(tensor)
                
                # Stack all samples: (n_samples, ...)
                stacked = np.stack(padded_tensors, axis=0)
                mean_tensor = stacked.mean(axis=0)

                if mean_tensor.ndim == 4:
                    # Shape: (batch, n_heads, seq, seq)
                    n_heads = mean_tensor.shape[1]
                    for head_idx in range(n_heads):
                        head_attn = mean_tensor[0, head_idx]
                        if head_idx not in self._head_means[axis_name][layer_name]:
                            self._head_means[axis_name][layer_name][head_idx] = {}
                        self._head_means[axis_name][layer_name][head_idx][group_name] = (
                            head_attn.flatten()
                        )
                else:
                    # Non-attention-weight layer — use as single "head 0"
                    if 0 not in self._head_means[axis_name][layer_name]:
                        self._head_means[axis_name][layer_name][0] = {}
                    self._head_means[axis_name][layer_name][0][group_name] = (
                        mean_tensor.flatten()
                    )

        self._hook_engine.detach()
        self._fitted_axes.append(axis_name)
        logger.info(
            f"HeadAttributor fit complete for axis '{axis_name}'"
        )

        return self


    def top_heads(
        self,
        axis_name: str,
        top_k: int = 5,
    ) -> List[Tuple[str, int, float]]:
        """
        Top-k attention heads most specialized for this axis.

        Specialization = how different the attention pattern is
        across groups (measured by variance of means).

        Args:
            axis_name: e.g. "language"
            top_k:     how many heads to return

        Returns:
            [(layer_name, head_idx, specialization_score), ...]
        """
        self._check_fitted(axis_name)

        scores = []

        for layer_name, heads in self._head_means[axis_name].items():
            for head_idx, group_means in heads.items():
                if len(group_means) < 2:
                    continue

                # Variance across group mean vectors
                # High variance = head behaves differently per group
                all_means = np.stack(list(group_means.values()))
                score = float(all_means.var(axis=0).mean())
                scores.append((layer_name, head_idx, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]


    def specialization_matrix(
        self,
        axis_name: str,
    ) -> Tuple[List[str], List[int], np.ndarray]:

        self._check_fitted(axis_name)

        layers = sorted(self._head_means[axis_name].keys())
        max_heads = max(
            max(heads.keys()) + 1
            for heads in self._head_means[axis_name].values()
            if heads
        )
        head_indices = list(range(max_heads))

        matrix = np.zeros((len(layers), max_heads))
        l_idx = {l: i for i, l in enumerate(layers)}

        for layer_name, heads in self._head_means[axis_name].items():
            for head_idx, group_means in heads.items():
                if len(group_means) < 2:
                    continue
                all_means = np.stack(list(group_means.values()))
                score = float(all_means.var(axis=0).mean())
                matrix[l_idx[layer_name], head_idx] = score

        return layers, head_indices, matrix


    def show(
        self,
        axis_name: str,
        top_k: int = 10,
    ) -> None:

        top = self.top_heads(axis_name, top_k=top_k)

        if not top:
            console.print("[yellow]No attention heads found.[/yellow]")
            return

        console.print()
        console.print(Panel(
            f"[dim]Axis:[/dim] [bold cyan]{axis_name}[/bold cyan]   "
            f"[dim]Top:[/dim] [white]{top_k} heads[/white]",
            title="[bold cyan]Attention Head Specialization[/bold cyan]",
            border_style="cyan",
            padding=(0, 4),
        ))

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 2),
            expand=True,
        )

        table.add_column("#",             width=4,  justify="right")
        table.add_column("Layer",         ratio=3,  style="white")
        table.add_column("Head",          width=6,  justify="center")
        table.add_column("Specialization",width=16, justify="right")
        table.add_column("Signal",        ratio=2)

        max_score = top[0][2] if top else 1.0

        for rank, (layer, head_idx, score) in enumerate(top, 1):
            ratio   = score / max_score if max_score > 0 else 0
            bar_len = int(ratio * 24)

            if ratio >= 0.7:
                style = "bold green"
            elif ratio >= 0.4:
                style = "bold yellow"
            else:
                style = "dim white"

            bar = Text()
            bar.append("█" * bar_len,        style=style)
            bar.append("░" * (24 - bar_len),  style="dim")

            short = (
                ".".join(layer.split(".")[-2:])
                if "." in layer else layer
            )

            table.add_row(
                str(rank),
                short,
                Text(f"H{head_idx}", style="bold magenta"),
                Text(f"{score:.6f}", style=style),
                bar,
            )

        console.print(table)
        console.print()


    def plot_heatmap(self, axis_name: str) -> None:

        try:
            import plotly.graph_objects as go
        except ImportError:
            console.print("[yellow]Plotly required.[/yellow]")
            return

        layers, heads, matrix = self.specialization_matrix(axis_name)

        short_layers = [
            ".".join(l.split(".")[-2:]) if "." in l else l
            for l in layers
        ]

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"H{h}" for h in heads],
            y=short_layers,
            colorscale="Plasma",
            colorbar=dict(title="Specialization"),
            hovertemplate=(
                "<b>Layer:</b> %{y}<br>"
                "<b>Head:</b> %{x}<br>"
                "<b>Score:</b> %{z:.6f}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=(
                f"Attention Head Specialization Map<br>"
                f"<sub>Axis: {axis_name} | "
                f"Model: {self._probe.model_name}</sub>"
            ),
            xaxis_title="Attention Head",
            yaxis_title="Layer",
            height=max(500, 22 * len(layers)),
            template="plotly_dark",
            font=dict(family="monospace"),
        )

        fig.show()


    def _check_fitted(self, axis_name: str) -> None:
        if axis_name not in self._fitted_axes:
            raise ValueError(
                f"Axis '{axis_name}' not fitted. "
                f"Call heads.fit('{axis_name}') first."
            )

    def __repr__(self) -> str:
        return (
            f"HeadAttributor("
            f"fitted_axes={self._fitted_axes})"
        )