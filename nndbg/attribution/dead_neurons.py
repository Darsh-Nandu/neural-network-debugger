"""
DeadNeuronDetector — identifies neurons that never (or rarely) activate.

A "dead" neuron is one whose activation is at or near zero across all
inputs, all groups, and all samples for a given layer.  This is a
well-known failure mode in networks that use ReLU-family activations
(the "dying ReLU" problem) but can also surface in any layer as a sign
of over-regularisation, poor initialisation, or representational collapse.

Three tiers:
    • dead        — fires < ``dead_threshold`` on every sample
    • near-dead   — mean absolute activation < ``near_dead_threshold``
    • saturated   — always fires at (or above) ``saturation_threshold``
                    (the "opposite" failure: always on, never selective)

Usage
-----
    detector = DeadNeuronDetector(probe)
    detector.fit(axis_name="language")        # re-uses already-run raw acts
    report   = detector.report("language")    # returns DeadNeuronReport
    report.show()                             # rich terminal table
    report.to_dict()                          # JSON-serialisable summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from nndbg.utils.console import console
from nndbg.utils.logging import get_logger

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

logger = get_logger(__name__)


@dataclass
class NeuronStatus:
    """Status of a single neuron inside one layer."""
    index: int
    mean_activation: float          # mean |activation| across all samples
    max_activation: float           # max  |activation| across all samples
    min_activation: float           # min  |activation| across all samples
    std_activation: float           # std  |activation| across all samples
    fire_rate: float                # fraction of samples where |act| > eps
    is_dead: bool
    is_near_dead: bool
    is_saturated: bool

    @property
    def status_label(self) -> str:
        if self.is_dead:
            return "dead"
        if self.is_near_dead:
            return "near-dead"
        if self.is_saturated:
            return "saturated"
        return "healthy"


@dataclass
class LayerDeadReport:
    """Dead-neuron summary for one layer."""
    layer_name: str
    n_neurons: int
    dead: List[NeuronStatus]        = field(default_factory=list)
    near_dead: List[NeuronStatus]   = field(default_factory=list)
    saturated: List[NeuronStatus]   = field(default_factory=list)
    healthy: List[NeuronStatus]     = field(default_factory=list)

    @property
    def n_dead(self) -> int:       return len(self.dead)
    @property
    def n_near_dead(self) -> int:  return len(self.near_dead)
    @property
    def n_saturated(self) -> int:  return len(self.saturated)
    @property
    def n_healthy(self) -> int:    return len(self.healthy)

    @property
    def all_neurons(self) -> List[NeuronStatus]:
        return self.dead + self.near_dead + self.saturated + self.healthy

    @property
    def layer_activation_score(self) -> float:
        activations = [n.mean_activation for n in self.all_neurons]
        return float(np.mean(activations)) if activations else 0.0

    @property
    def dead_fraction(self) -> float:
        return self.n_dead / max(self.n_neurons, 1)

    @property
    def near_dead_fraction(self) -> float:
        return self.n_near_dead / max(self.n_neurons, 1)

    @property
    def health_score(self) -> float:
        """0.0 = entirely dead, 1.0 = entirely healthy."""
        problem = self.n_dead + 0.5 * self.n_near_dead + 0.3 * self.n_saturated
        return max(0.0, 1.0 - problem / max(self.n_neurons, 1))


@dataclass
class DeadNeuronReport:
    """Full dead-neuron report across all layers for one axis."""
    axis_name: str
    model_name: str
    layers: Dict[str, LayerDeadReport]   = field(default_factory=dict)

    # thresholds used during analysis (stored for reproducibility)
    dead_threshold: float = 1e-6
    near_dead_threshold: float = 0.01
    saturation_threshold: float = 0.95   # fraction of max range

    def worst_layers(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Layers ranked by dead_fraction (worst first)."""
        ranked = sorted(
            [(name, r.dead_fraction) for name, r in self.layers.items()],
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]

    def top_active_layers(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Layers ranked by average per-neuron mean activation (best first)."""
        ranked = sorted(
            [(name, r.layer_activation_score) for name, r in self.layers.items()],
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]

    def top_active_neurons(self, top_k: int = 20) -> List[Tuple[str, NeuronStatus]]:
        """Neurons ranked by mean absolute activation across all samples."""
        all_neurons = [
            (layer_name, ns)
            for layer_name, layer_report in self.layers.items()
            for ns in layer_report.all_neurons
        ]
        ranked = sorted(all_neurons, key=lambda x: x[1].mean_activation, reverse=True)
        return ranked[:top_k]

    def total_dead(self) -> int:
        return sum(r.n_dead for r in self.layers.values())

    def total_near_dead(self) -> int:
        return sum(r.n_near_dead for r in self.layers.values())

    def total_neurons(self) -> int:
        return sum(r.n_neurons for r in self.layers.values())

    def global_dead_fraction(self) -> float:
        return self.total_dead() / max(self.total_neurons(), 1)

    def show(
        self,
        top_k: int = 15,
        show_neurons: bool = False,
        min_dead_fraction: float = 0.0,
    ) -> None:
        """Print a rich terminal report."""

        header = Text()
        header.append("Model    ", style="dim")
        header.append(f"{self.model_name}\n", style="bold white")
        header.append("Axis     ", style="dim")
        header.append(f"{self.axis_name}\n", style="bold cyan")
        header.append("Total    ", style="dim")
        header.append(
            f"{self.total_neurons()} neurons  |  "
            f"{self.total_dead()} dead  "
            f"({self.global_dead_fraction():.1%})  |  "
            f"{self.total_near_dead()} near-dead",
            style="white",
        )

        console.print()
        console.print(Panel(
            header,
            title="[bold red]Dead Neuron Analysis[/bold red]",
            border_style="red",
            padding=(1, 4),
        ))

        filtered = {
            name: r for name, r in self.layers.items()
            if r.dead_fraction >= min_dead_fraction
        }
        # sort worst-first, cap at top_k
        sorted_layers = sorted(
            filtered.items(),
            key=lambda x: x[1].dead_fraction,
            reverse=True,
        )[:top_k]

        if not sorted_layers:
            console.print("[green]  No dead neurons found above threshold.[/green]")
        else:
            table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
                expand=True,
            )
            table.add_column("#",           style="dim",          width=4,  justify="right")
            table.add_column("Layer",       style="white",        ratio=4)
            table.add_column("Neurons",     style="white",        width=9,  justify="right")
            table.add_column("Dead",        style="bold red",     width=9,  justify="right")
            table.add_column("Near-dead",   style="bold yellow",  width=10, justify="right")
            table.add_column("Saturated",   style="bold magenta", width=10, justify="right")
            table.add_column("Health",      ratio=2)

            for rank, (layer_name, report) in enumerate(sorted_layers, 1):
                hs = report.health_score
                filled = int(hs * 24)

                if hs >= 0.8:
                    bar_style = "bold green"
                elif hs >= 0.5:
                    bar_style = "bold yellow"
                else:
                    bar_style = "bold red"

                bar = Text()
                bar.append("█" * filled, style=bar_style)
                bar.append("░" * (24 - filled), style="dim")
                bar.append(f"  {hs:.1%}", style=bar_style)

                short = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name

                table.add_row(
                    str(rank),
                    short,
                    str(report.n_neurons),
                    f"{report.n_dead} ({report.dead_fraction:.0%})",
                    f"{report.n_near_dead} ({report.near_dead_fraction:.0%})",
                    str(report.n_saturated),
                    bar,
                )

            console.print(table)

        top_layers = self.top_active_layers(top_k=5)
        if top_layers:
            active_table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold green",
                padding=(0, 2),
                expand=True,
            )
            active_table.add_column("Rank", width=4, justify="right")
            active_table.add_column("Layer", style="white", ratio=4)
            active_table.add_column("Avg mean |act|", style="bold green", width=16, justify="right")

            for rank, (layer_name, score) in enumerate(top_layers, 1):
                short = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name
                active_table.add_row(str(rank), short, f"{score:.2e}")

            console.print(Panel(
                active_table,
                title="[bold green]Top active layers[/bold green]",
                border_style="green",
                padding=(1, 1),
            ))

        top_neurons = self.top_active_neurons(top_k=10)
        if top_neurons:
            neuron_table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold green",
                padding=(0, 2),
                expand=True,
            )
            neuron_table.add_column("Rank", width=4, justify="right")
            neuron_table.add_column("Layer", style="white", ratio=3)
            neuron_table.add_column("Neuron", width=9, justify="right")
            neuron_table.add_column("Status", width=10)
            neuron_table.add_column("Mean |act|", width=14, justify="right")
            neuron_table.add_column("Fire rate", width=10, justify="right")

            for rank, (layer_name, ns) in enumerate(top_neurons, 1):
                short = ".".join(layer_name.split(".")[-2:]) if "." in layer_name else layer_name
                neuron_table.add_row(
                    str(rank),
                    short,
                    f"#{ns.index}",
                    ns.status_label,
                    f"{ns.mean_activation:.2e}",
                    f"{ns.fire_rate:.1%}",
                )

            console.print(Panel(
                neuron_table,
                title="[bold green]Most activated neurons[/bold green]",
                border_style="green",
                padding=(1, 1),
            ))

        if show_neurons:
            for layer_name, report in sorted_layers:
                if not report.dead and not report.near_dead:
                    continue
                console.print(
                    f"\n  [bold white]{layer_name}[/bold white]  "
                    f"[dim]{report.n_dead} dead, {report.n_near_dead} near-dead[/dim]"
                )
                nt = Table(box=box.SIMPLE, show_header=True,
                           header_style="bold cyan", padding=(0, 2))
                nt.add_column("Neuron",   width=9, justify="right")
                nt.add_column("Status",   width=10)
                nt.add_column("Mean |act|", width=12, justify="right")
                nt.add_column("Max |act|",  width=12, justify="right")
                nt.add_column("Fire rate",  width=10, justify="right")

                all_bad = sorted(
                    report.dead + report.near_dead,
                    key=lambda n: n.mean_activation,
                )
                for ns in all_bad:
                    status_style = "bold red" if ns.is_dead else "bold yellow"
                    nt.add_row(
                        f"#{ns.index}",
                        Text(ns.status_label, style=status_style),
                        f"{ns.mean_activation:.2e}",
                        f"{ns.max_activation:.2e}",
                        f"{ns.fire_rate:.1%}",
                    )
                console.print(nt)

        console.print()


    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "axis_name":             self.axis_name,
            "model_name":            self.model_name,
            "thresholds": {
                "dead":         self.dead_threshold,
                "near_dead":    self.near_dead_threshold,
                "saturation":   self.saturation_threshold,
            },
            "global_summary": {
                "total_neurons":        self.total_neurons(),
                "total_dead":           self.total_dead(),
                "total_near_dead":      self.total_near_dead(),
                "global_dead_fraction": self.global_dead_fraction(),
            },
            "layers": {
                name: {
                    "n_neurons":         r.n_neurons,
                    "n_dead":            r.n_dead,
                    "n_near_dead":       r.n_near_dead,
                    "n_saturated":       r.n_saturated,
                    "n_healthy":         r.n_healthy,
                    "dead_fraction":     r.dead_fraction,
                    "near_dead_fraction":r.near_dead_fraction,
                    "health_score":      r.health_score,
                    "avg_activation":    r.layer_activation_score,
                    "dead_neurons": [
                        {
                            "index":            ns.index,
                            "mean_activation":  ns.mean_activation,
                            "max_activation":   ns.max_activation,
                            "min_activation":   ns.min_activation,
                            "std_activation":   ns.std_activation,
                            "fire_rate":        ns.fire_rate,
                        }
                        for ns in r.dead
                    ],
                    "near_dead_neurons": [
                        {
                            "index":            ns.index,
                            "mean_activation":  ns.mean_activation,
                            "max_activation":   ns.max_activation,
                            "min_activation":   ns.min_activation,
                            "std_activation":   ns.std_activation,
                            "fire_rate":        ns.fire_rate,
                        }
                        for ns in r.near_dead
                    ],
                    "saturated_neurons": [
                        {
                            "index":   ns.index,
                            "mean_activation": ns.mean_activation,
                        }
                        for ns in r.saturated
                    ],
                }
                for name, r in self.layers.items()
            },
            "top_active_layers": [
                {
                    "layer_name": name,
                    "avg_mean_activation": score,
                }
                for name, score in self.top_active_layers(top_k=10)
            ],
            "top_active_neurons": [
                {
                    "layer_name": layer_name,
                    "index": ns.index,
                    "status": ns.status_label,
                    "mean_activation": ns.mean_activation,
                    "max_activation": ns.max_activation,
                    "fire_rate": ns.fire_rate,
                }
                for layer_name, ns in self.top_active_neurons(top_k=20)
            ],
        }


class DeadNeuronDetector:
    """
    Detects dead, near-dead, and saturated neurons across every layer
    captured during a ModelProbe run.

    Parameters
    ----------
    probe : ModelProbe
        A configured (and optionally already-run) ModelProbe instance.
    dead_threshold : float
        Neurons whose max |activation| across ALL samples is below this
        value are classified as dead.  Default: 1e-6.
    near_dead_threshold : float
        Neurons whose mean |activation| is below this value (but not
        fully dead) are classified as near-dead.  Default: 0.01.
    saturation_threshold : float
        Neurons whose fire_rate is above this value are classified as
        saturated (always-on).  Default: 0.95.
    eps : float
        A small value used to decide if a single sample "fired" at all.
        Default: 1e-8.
    """

    def __init__(
        self,
        probe,
        dead_threshold: float = 1e-6,
        near_dead_threshold: float = 0.01,
        saturation_threshold: float = 0.95,
        eps: float = 1e-8,
    ):
        self._probe = probe
        self.dead_threshold = dead_threshold
        self.near_dead_threshold = near_dead_threshold
        self.saturation_threshold = saturation_threshold
        self.eps = eps

        # axis_name -> DeadNeuronReport
        self._reports: Dict[str, DeadNeuronReport] = {}

    def _is_neuron_layer(self, layer_name: str) -> bool:
        """Skip captured modules that are pure activations/normalisation/dropout."""
        engine = getattr(self._probe, "_hook_engine", None)
        if engine is None:
            return True
        layer = getattr(engine, "_layer_registry", {}).get(layer_name)
        if layer is None:
            return True

        excluded = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.ReLU,
            nn.GELU,
            nn.Sigmoid,
            nn.Tanh,
            nn.Softmax,
            nn.LogSoftmax,
            nn.Softplus,
            nn.SiLU,
            nn.LeakyReLU,
            nn.PReLU,
            nn.ELU,
            nn.SELU,
            nn.Hardtanh,
            nn.Softsign,
            nn.Softmin,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.Flatten,
            nn.Identity,
            nn.Sequential,
        )
        return not isinstance(layer, excluded)


    def fit(
        self,
        axis_name: str,
        raw_activations: Optional[Dict] = None,
    ) -> "DeadNeuronDetector":

        # layer_name -> list of 1-D float arrays (one per sample, all groups merged)
        layer_all_samples: Dict[str, List[np.ndarray]] = {}

        if raw_activations and axis_name in raw_activations:
            # re-use already-collected tensors
            axis_raws = raw_activations[axis_name]
            for group_name, samples in axis_raws.items():
                for sample_idx, layer_tensors in samples.items():
                    for layer_name, tensor in layer_tensors.items():
                        if not self._is_neuron_layer(layer_name):
                            logger.debug(f"Skipping non-neuron layer '{layer_name}'")
                            continue
                        vec = self._to_neuron_vec(tensor)
                        if vec is None:
                            continue
                        layer_all_samples.setdefault(layer_name, []).append(vec)
        else:
            # run fresh forward passes
            axis = next(
                (a for a in self._probe._axes if a.name == axis_name), None
            )
            if axis is None:
                raise ValueError(
                    f"Axis '{axis_name}' not found. "
                    f"Available: {[a.name for a in self._probe._axes]}"
                )
            self._probe._hook_engine.attach()
            for group_name, texts in axis.groups.items():
                for text in texts:
                    activations = self._probe._run_single(text)
                    for layer_name, tensor in activations.items():
                        if not self._is_neuron_layer(layer_name):
                            logger.debug(f"Skipping non-neuron layer '{layer_name}'")
                            continue
                        vec = self._to_neuron_vec(tensor)
                        if vec is None:
                            continue
                        layer_all_samples.setdefault(layer_name, []).append(vec)
            self._probe._hook_engine.detach()

        if not layer_all_samples:
            logger.warning(
                f"DeadNeuronDetector: no activations found for axis '{axis_name}'"
            )
            return self

        report = DeadNeuronReport(
            axis_name=axis_name,
            model_name=self._probe.model_name,
            dead_threshold=self.dead_threshold,
            near_dead_threshold=self.near_dead_threshold,
            saturation_threshold=self.saturation_threshold,
        )

        for layer_name, sample_vecs in layer_all_samples.items():
            n_neurons = sample_vecs[0].shape[0]
            # Stack → shape (n_samples, n_neurons)
            matrix = np.stack(sample_vecs, axis=0)            # (S, N)
            abs_matrix = np.abs(matrix)

            mean_act  = abs_matrix.mean(axis=0)               # (N,)
            max_act   = abs_matrix.max(axis=0)                # (N,)
            min_act   = abs_matrix.min(axis=0)                # (N,)
            std_act   = abs_matrix.std(axis=0)                # (N,)
            fire_rate = (abs_matrix > self.eps).mean(axis=0)  # (N,)

            layer_report = LayerDeadReport(
                layer_name=layer_name,
                n_neurons=n_neurons,
            )

            for idx in range(n_neurons):
                ns = NeuronStatus(
                    index=idx,
                    mean_activation=float(mean_act[idx]),
                    max_activation=float(max_act[idx]),
                    min_activation=float(min_act[idx]),
                    std_activation=float(std_act[idx]),
                    fire_rate=float(fire_rate[idx]),
                    is_dead=bool(max_act[idx] < self.dead_threshold),
                    is_near_dead=bool(
                        max_act[idx] >= self.dead_threshold
                        and mean_act[idx] < self.near_dead_threshold
                    ),
                    is_saturated=bool(fire_rate[idx] >= self.saturation_threshold),
                )

                if ns.is_dead:
                    layer_report.dead.append(ns)
                elif ns.is_near_dead:
                    layer_report.near_dead.append(ns)
                elif ns.is_saturated:
                    layer_report.saturated.append(ns)
                else:
                    layer_report.healthy.append(ns)

            report.layers[layer_name] = layer_report
            logger.debug(
                f"  {layer_name}: {layer_report.n_dead} dead, "
                f"{layer_report.n_near_dead} near-dead, "
                f"{layer_report.n_saturated} saturated "
                f"/ {n_neurons} total"
            )

        self._reports[axis_name] = report
        logger.info(
            f"DeadNeuronDetector fit complete — axis='{axis_name}', "
            f"global dead fraction={report.global_dead_fraction():.1%}"
        )
        return self

    def report(self, axis_name: str) -> DeadNeuronReport:
        """Return the report for a fitted axis."""
        if axis_name not in self._reports:
            raise ValueError(
                f"Axis '{axis_name}' not yet analysed. "
                f"Call detector.fit('{axis_name}') first."
            )
        return self._reports[axis_name]

    def fit_all(
        self,
        raw_activations: Optional[Dict] = None,
    ) -> "DeadNeuronDetector":
        """Fit all axes attached to the probe."""
        for axis in self._probe._axes:
            self.fit(axis.name, raw_activations=raw_activations)
        return self
    

    @staticmethod
    def _to_neuron_vec(tensor) -> Optional[np.ndarray]:
        """
        Collapse a tensor to a 1-D array of shape (n_neurons,).

        Strategy:
        - 1-D tensor (n,)         → already per-neuron
        - 2-D tensor (seq, n)     → mean over seq dim
        - 3-D tensor (b, seq, n)  → mean over b and seq
        - 4-D+                    → mean over all but last dim
        - scalar / 0-D            → skip (return None)
        """
        import torch
        if isinstance(tensor, torch.Tensor):
            arr = tensor.float().detach().cpu().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            return arr
        # peak-pool over all dims except the last so sparse sequence activations
        # are not averaged away.
        return np.max(np.abs(arr), axis=tuple(range(arr.ndim - 1))).flatten()

    def __repr__(self) -> str:
        fitted = list(self._reports.keys())
        return f"DeadNeuronDetector(fitted_axes={fitted})"