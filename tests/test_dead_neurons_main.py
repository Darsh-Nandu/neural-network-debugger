"""
test_dead_neurons.py — Full test suite for DeadNeuronDetector.

Covers:
  1.  NeuronStatus dataclass — labels, status flags
  2.  LayerDeadReport — counts, fractions, health_score
  3.  DeadNeuronReport — aggregation, worst_layers, to_dict structure
  4.  DeadNeuronDetector construction — threshold params stored correctly
  5.  DeadNeuronDetector._to_neuron_vec — all tensor shapes handled
  6.  DeadNeuronDetector.fit() from raw_activations — happy path
  7.  DeadNeuronDetector.fit() from fresh forward passes — tiny model
  8.  Detection correctness — known-dead / near-dead / saturated / healthy
  9.  Edge cases — all dead, all healthy, single neuron, empty axis
  10. fit_all() — fits every axis on the probe
  11. report() — raises if axis not fitted
  12. save() integration — dead_neurons JSON in zip, neuron_raw CSVs present
  13. DeadNeuronReport.show() — smoke-test (no crash)

Run with:
    PYTHONPATH=. pytest test_dead_neurons.py -v
"""

from __future__ import annotations

import io
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from nndbg import ModelProbe, DeadNeuronDetector, DeadNeuronReport, set_verbose
from nndbg.attribution.dead_neurons import (
    NeuronStatus,
    LayerDeadReport,
)
from nndbg.probing.axis import Axis
from nndbg.results import ProbeResults
from nndbg.storage.store import ActivationStore


class TinyModel(nn.Module):
    """2-layer MLP with ReLU — simple enough to produce genuine dead neurons."""

    def __init__(self, in_dim: int = 16, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ZeroModel(nn.Module):
    """Model whose hidden layer always outputs zero — all neurons are dead."""

    def __init__(self, in_dim: int = 8, hidden: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        nn.init.zeros_(self.fc1.weight)   # weight = 0  → output always 0
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class DummyTokenizer:
    """Returns deterministic float tensors — no real tokeniser needed."""

    def __init__(self, in_dim: int = 16):
        self.in_dim = in_dim

    def __call__(self, text, return_tensors="pt", max_length=128,
                 truncation=True, padding=True):
        seed = len(text) % 100
        torch.manual_seed(seed)
        return {"x": torch.randn(1, self.in_dim)}


@pytest.fixture()
def tiny_probe():
    set_verbose(False)
    model = TinyModel()
    tok   = DummyTokenizer()
    probe = ModelProbe.from_model(model, tokenizer=tok, model_name="tiny")
    probe.add_axis("sentiment", {
        "positive": ["good", "great", "excellent"],
        "negative": ["bad",  "awful", "terrible"],
    })
    return probe


@pytest.fixture()
def zero_probe():
    """Probe whose fc1 layer will always produce dead activations."""
    set_verbose(False)
    model = ZeroModel()
    tok   = DummyTokenizer(in_dim=8)
    probe = ModelProbe.from_model(model, tokenizer=tok, model_name="zero")
    probe.add_axis("dummy", {
        "A": ["x", "y", "z"],
        "B": ["a", "b", "c"],
    })
    return probe


def _make_raw_activations(
    axis_name: str = "sentiment",
    groups: tuple = ("positive", "negative"),
    n_samples: int = 3,
    layer_name: str = "fc1",
    tensor_factory=None,
) -> Dict:
    """Build a synthetic raw_activations dict like ModelProbe.run() produces."""
    if tensor_factory is None:
        tensor_factory = lambda g, s: torch.randn(1, 8)

    return {
        axis_name: {
            group: {
                i: {layer_name: tensor_factory(group, i)}
                for i in range(n_samples)
            }
            for group in groups
        }
    }


def _make_probe_results_with_raw(raw_activations, axis_name="sentiment"):
    axes = [Axis(name=axis_name, groups={
        g: [f"text_{i}" for i in range(len(samples))]
        for g, samples in raw_activations[axis_name].items()
    })]
    store = ActivationStore(":memory:")
    return ProbeResults(
        run_id="dead001",
        model_name="test-model",
        axes=axes,
        probe_scores={axis_name: {"fc1": 0.7}},
        store=store,
        layer_group_data={axis_name: {"fc1": {"positive": [], "negative": []}}},
        raw_activations=raw_activations,
    )


class TestNeuronStatus:

    def _make(self, is_dead=False, is_near_dead=False, is_saturated=False):
        return NeuronStatus(
            index=0,
            mean_activation=0.0,
            max_activation=0.0,
            min_activation=0.0,
            std_activation=0.0,
            fire_rate=0.0,
            is_dead=is_dead,
            is_near_dead=is_near_dead,
            is_saturated=is_saturated,
        )

    def test_dead_label(self):
        assert self._make(is_dead=True).status_label == "dead"

    def test_near_dead_label(self):
        assert self._make(is_near_dead=True).status_label == "near-dead"

    def test_saturated_label(self):
        assert self._make(is_saturated=True).status_label == "saturated"

    def test_healthy_label(self):
        assert self._make().status_label == "healthy"

    def test_dead_takes_priority_over_near_dead(self):
        # A dead neuron should report as dead even if near_dead is also set
        ns = self._make(is_dead=True, is_near_dead=True)
        assert ns.status_label == "dead"

    def test_index_stored(self):
        ns = NeuronStatus(
            index=42, mean_activation=0.1, max_activation=0.5,
            min_activation=0.0, std_activation=0.05, fire_rate=0.8,
            is_dead=False, is_near_dead=False, is_saturated=False,
        )
        assert ns.index == 42



class TestLayerDeadReport:

    def _make_report(self, n_dead=2, n_near_dead=3, n_saturated=1, n_healthy=4):
        r = LayerDeadReport(layer_name="fc1", n_neurons=n_dead+n_near_dead+n_saturated+n_healthy)

        def _ns(**kwargs):
            return NeuronStatus(
                index=0, mean_activation=0.0, max_activation=0.0,
                min_activation=0.0, std_activation=0.0, fire_rate=0.0,
                **kwargs,
            )

        r.dead      = [_ns(is_dead=True,       is_near_dead=False, is_saturated=False)] * n_dead
        r.near_dead = [_ns(is_dead=False,       is_near_dead=True,  is_saturated=False)] * n_near_dead
        r.saturated = [_ns(is_dead=False,       is_near_dead=False, is_saturated=True)]  * n_saturated
        r.healthy   = [_ns(is_dead=False,       is_near_dead=False, is_saturated=False)] * n_healthy
        return r

    def test_counts(self):
        r = self._make_report(n_dead=2, n_near_dead=3, n_saturated=1, n_healthy=4)
        assert r.n_dead      == 2
        assert r.n_near_dead == 3
        assert r.n_saturated == 1
        assert r.n_healthy   == 4

    def test_dead_fraction(self):
        r = self._make_report(n_dead=2, n_near_dead=0, n_saturated=0, n_healthy=8)
        assert pytest.approx(r.dead_fraction, abs=1e-6) == 0.2

    def test_near_dead_fraction(self):
        r = self._make_report(n_dead=0, n_near_dead=5, n_saturated=0, n_healthy=5)
        assert pytest.approx(r.near_dead_fraction, abs=1e-6) == 0.5

    def test_health_score_all_healthy(self):
        r = self._make_report(n_dead=0, n_near_dead=0, n_saturated=0, n_healthy=10)
        assert pytest.approx(r.health_score, abs=1e-6) == 1.0

    def test_health_score_all_dead(self):
        r = self._make_report(n_dead=10, n_near_dead=0, n_saturated=0, n_healthy=0)
        assert r.health_score == 0.0

    def test_health_score_in_range(self):
        r = self._make_report(n_dead=2, n_near_dead=3, n_saturated=1, n_healthy=4)
        assert 0.0 <= r.health_score <= 1.0

    def test_zero_neurons_no_division_error(self):
        r = LayerDeadReport(layer_name="empty", n_neurons=0)
        assert r.dead_fraction == 0.0
        assert r.health_score == 1.0



class TestDeadNeuronReport:

    def _make_report(self):
        r = DeadNeuronReport(axis_name="sentiment", model_name="test")

        def _layer(name, n_dead, n_neurons, healthy_mean):
            lr = LayerDeadReport(layer_name=name, n_neurons=n_neurons)
            lr.dead = [
                NeuronStatus(
                    index=i, mean_activation=0.0, max_activation=0.0,
                    min_activation=0.0, std_activation=0.0, fire_rate=0.0,
                    is_dead=True, is_near_dead=False, is_saturated=False,
                )
                for i in range(n_dead)
            ]
            lr.healthy = [
                NeuronStatus(
                    index=i, mean_activation=healthy_mean, max_activation=1.0,
                    min_activation=0.0, std_activation=0.1, fire_rate=0.9,
                    is_dead=False, is_near_dead=False, is_saturated=False,
                )
                for i in range(n_neurons - n_dead)
            ]
            return lr

        r.layers = {
            "fc1": _layer("fc1", n_dead=3, n_neurons=10, healthy_mean=0.5),
            "fc2": _layer("fc2", n_dead=1, n_neurons=5, healthy_mean=0.7),
            "fc3": _layer("fc3", n_dead=0, n_neurons=8, healthy_mean=0.3),
        }
        return r

    def test_total_dead(self):
        r = self._make_report()
        assert r.total_dead() == 4

    def test_total_neurons(self):
        r = self._make_report()
        assert r.total_neurons() == 23

    def test_global_dead_fraction(self):
        r = self._make_report()
        assert pytest.approx(r.global_dead_fraction(), abs=1e-4) == 4 / 23

    def test_worst_layers_order(self):
        r = self._make_report()
        worst = r.worst_layers(top_k=3)
        fracs = [f for _, f in worst]
        assert fracs == sorted(fracs, reverse=True)

    def test_worst_layers_top_k(self):
        r = self._make_report()
        assert len(r.worst_layers(top_k=2)) == 2

    def test_to_dict_keys(self):
        r = self._make_report()
        d = r.to_dict()
        for key in ("axis_name", "model_name", "thresholds", "global_summary", "layers"):
            assert key in d

    def test_to_dict_global_summary(self):
        r = self._make_report()
        gs = r.to_dict()["global_summary"]
        assert gs["total_dead"] == 4
        assert gs["total_neurons"] == 23

    def test_to_dict_layer_keys(self):
        r = self._make_report()
        layers_d = r.to_dict()["layers"]
        assert set(layers_d.keys()) == {"fc1", "fc2", "fc3"}

    def test_to_dict_dead_neuron_list(self):
        r = self._make_report()
        fc1 = r.to_dict()["layers"]["fc1"]
        assert len(fc1["dead_neurons"]) == 3
        assert "index" in fc1["dead_neurons"][0]
        assert "mean_activation" in fc1["dead_neurons"][0]

    def test_top_active_layers_order(self):
        r = self._make_report()
        top = r.top_active_layers(top_k=3)
        assert [name for name, _ in top] == ["fc2", "fc1", "fc3"]

    def test_top_active_neurons(self):
        r = self._make_report()
        top = r.top_active_neurons(top_k=1)
        assert top[0][0] == "fc2"
        assert top[0][1].mean_activation == pytest.approx(0.7)

    def test_to_dict_includes_top_active(self):
        r = self._make_report()
        d = r.to_dict()
        assert "top_active_layers" in d
        assert "top_active_neurons" in d
        assert d["top_active_layers"][0]["layer_name"] == "fc2"

    def test_to_dict_is_json_serialisable(self):
        r = self._make_report()
        json.dumps(r.to_dict())   # must not raise


class TestDeadNeuronDetectorConstruction:

    def test_default_thresholds(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        assert d.dead_threshold      == 1e-6
        assert d.near_dead_threshold == 0.01
        assert d.saturation_threshold == 0.95

    def test_custom_thresholds_stored(self, tiny_probe):
        d = DeadNeuronDetector(
            tiny_probe,
            dead_threshold=1e-4,
            near_dead_threshold=0.05,
            saturation_threshold=0.9,
        )
        assert d.dead_threshold       == 1e-4
        assert d.near_dead_threshold  == 0.05
        assert d.saturation_threshold == 0.9

    def test_repr_shows_no_fitted_axes_initially(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        assert "[]" in repr(d)

    def test_report_before_fit_raises(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        with pytest.raises(ValueError, match="not yet analysed"):
            d.report("sentiment")


class TestToNeuronVec:

    def _call(self, t):
        return DeadNeuronDetector._to_neuron_vec(t)

    def test_1d_tensor_passthrough(self):
        t = torch.tensor([0.1, 0.2, 0.3])
        v = self._call(t)
        assert v.shape == (3,)
        assert pytest.approx(v.tolist(), abs=1e-5) == [0.1, 0.2, 0.3]

    def test_2d_tensor_collapses_to_last_dim(self):
        t = torch.ones(4, 6)
        v = self._call(t)
        assert v.shape == (6,)

    def test_3d_tensor_collapses_to_last_dim(self):
        t = torch.ones(2, 5, 8)
        v = self._call(t)
        assert v.shape == (8,)

    def test_sparse_3d_activation_uses_peak(self):
        t = np.zeros((1, 10, 4), dtype=np.float32)
        t[0, 5, 2] = 1.0
        v = self._call(t)
        assert pytest.approx(v[2], abs=1e-6) == 1.0
        assert v.shape == (4,)

    def test_4d_tensor_collapses_to_last_dim(self):
        t = torch.ones(2, 3, 4, 10)
        v = self._call(t)
        assert v.shape == (10,)

    def test_scalar_returns_none(self):
        t = torch.tensor(0.5)   # 0-D tensor
        assert self._call(t) is None

    def test_nan_sanitised(self):
        t = torch.tensor([float("nan"), 1.0, float("inf")])
        v = self._call(t)
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))

    def test_numpy_array_accepted(self):
        a = np.array([0.1, 0.2, 0.5], dtype=np.float32)
        v = self._call(a)
        assert v.shape == (3,)



class TestFitFromRawActivations:

    def test_fit_returns_self(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        ret = d.fit("sentiment", raw_activations=raw)
        assert ret is d

    def test_report_available_after_fit(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        report = d.report("sentiment")
        assert isinstance(report, DeadNeuronReport)

    def test_report_axis_name_correct(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        assert d.report("sentiment").axis_name == "sentiment"

    def test_report_model_name_correct(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        assert d.report("sentiment").model_name == "tiny"

    def test_layers_present_in_report(self, tiny_probe):
        raw = _make_raw_activations(layer_name="fc1")
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        assert "fc1" in d.report("sentiment").layers

    def test_n_neurons_matches_tensor_width(self, tiny_probe):
        # 8-neuron vectors
        raw = _make_raw_activations(
            layer_name="fc1",
            tensor_factory=lambda g, s: torch.randn(1, 8)
        )
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        assert d.report("sentiment").layers["fc1"].n_neurons == 8

    def test_all_neuron_counts_add_up(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r   = d.report("sentiment").layers["fc1"]
        assert r.n_dead + r.n_near_dead + r.n_saturated + r.n_healthy == r.n_neurons

    def test_repr_updated_after_fit(self, tiny_probe):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        assert "sentiment" in repr(d)


class TestFitFreshForwardPasses:

    def test_fit_without_raw_activations(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment")   # no raw_activations → runs forward passes
        assert "sentiment" in d._reports

    def test_layers_populated(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment")
        report = d.report("sentiment")
        assert len(report.layers) > 0

    def test_non_neuron_modules_skipped(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment")
        report = d.report("sentiment")
        assert "relu" not in report.layers
        assert "fc1" in report.layers
        assert "fc2" in report.layers

    def test_neuron_counts_consistent(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment")
        for layer_report in d.report("sentiment").layers.values():
            total = (
                layer_report.n_dead + layer_report.n_near_dead
                + layer_report.n_saturated + layer_report.n_healthy
            )
            assert total == layer_report.n_neurons

    def test_invalid_axis_raises(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        with pytest.raises(ValueError, match="not found"):
            d.fit("nonexistent_axis")


class TestDetectionCorrectness:

    def _fit_with_known_tensors(self, probe, values: list, n_neurons: int = 6):
        """
        Feed tensors whose neuron activations are fixed to `values`.
        values: list of floats, one per sample (same for all neurons).
        """
        raw = {
            "sentiment": {
                "positive": {
                    i: {"known_layer": torch.full((1, n_neurons), values[i])}
                    for i in range(len(values))
                },
                "negative": {
                    i: {"known_layer": torch.full((1, n_neurons), values[i])}
                    for i in range(len(values))
                },
            }
        }
        d = DeadNeuronDetector(probe, dead_threshold=1e-6, near_dead_threshold=0.01)
        d.fit("sentiment", raw_activations=raw)
        return d.report("sentiment").layers["known_layer"]

    def test_zero_activations_are_dead(self, tiny_probe):
        lr = self._fit_with_known_tensors(tiny_probe, [0.0, 0.0, 0.0])
        assert lr.n_dead == lr.n_neurons
        assert lr.n_near_dead == 0

    def test_tiny_activations_are_near_dead(self, tiny_probe):
        # 0.005 < near_dead_threshold=0.01 but > dead_threshold=1e-6
        lr = self._fit_with_known_tensors(tiny_probe, [0.005, 0.005, 0.005])
        assert lr.n_near_dead == lr.n_neurons
        assert lr.n_dead == 0

    def test_large_activations_are_not_dead_or_near_dead(self, tiny_probe):
        # Large values are not dead/near-dead; they fire on every sample
        # so are classified as saturated (always-on) — a separate concern.
        lr = self._fit_with_known_tensors(tiny_probe, [1.0, 2.0, 3.0])
        assert lr.n_dead      == 0
        assert lr.n_near_dead == 0

    def test_variable_activations_are_healthy(self, tiny_probe):
        # Alternating zero/nonzero → fire_rate=0.5 < saturation_threshold=0.95
        # mean activation > near_dead_threshold → classified as healthy.
        d = DeadNeuronDetector(tiny_probe, saturation_threshold=0.95)
        raw = {
            "sentiment": {
                "positive": {
                    0: {"var_layer": torch.tensor([[0.0, 1.5, 0.0, 1.5]])},
                    1: {"var_layer": torch.tensor([[1.5, 0.0, 1.5, 0.0]])},
                },
                "negative": {
                    0: {"var_layer": torch.tensor([[0.0, 1.5, 0.0, 1.5]])},
                    1: {"var_layer": torch.tensor([[1.5, 0.0, 1.5, 0.0]])},
                },
            }
        }
        d.fit("sentiment", raw_activations=raw)
        lr = d.report("sentiment").layers["var_layer"]
        assert lr.n_dead      == 0
        assert lr.n_near_dead == 0
        assert lr.n_saturated == 0
        assert lr.n_healthy   == 4

    def test_saturated_neurons_detected(self, tiny_probe):
        # All samples fire > eps → fire_rate = 1.0 ≥ saturation_threshold=0.95
        d = DeadNeuronDetector(tiny_probe, saturation_threshold=0.95)
        raw = {
            "sentiment": {
                "positive": {
                    i: {"sat_layer": torch.ones(1, 4) * 2.0}
                    for i in range(5)
                },
                "negative": {
                    i: {"sat_layer": torch.ones(1, 4) * 2.0}
                    for i in range(5)
                },
            }
        }
        d.fit("sentiment", raw_activations=raw)
        lr = d.report("sentiment").layers["sat_layer"]
        assert lr.n_saturated == lr.n_neurons

    def test_dead_neurons_on_zero_model(self, zero_probe):
        """ZeroModel's fc1 layer must have all dead neurons."""
        d = DeadNeuronDetector(zero_probe, dead_threshold=1e-6)
        d.fit("dummy")
        # fc1 weights are zero → fc1 output is zero → all neurons dead
        r = d.report("dummy")
        # Find fc1 layer in report
        fc1_reports = {k: v for k, v in r.layers.items() if "fc1" in k}
        assert fc1_reports, "fc1 layer not found in report"
        fc1 = list(fc1_reports.values())[0]
        assert fc1.n_dead == fc1.n_neurons


class TestEdgeCases:

    def test_single_neuron_layer(self, tiny_probe):
        raw = _make_raw_activations(
            tensor_factory=lambda g, s: torch.randn(1, 1)
        )
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r = d.report("sentiment").layers["fc1"]
        assert r.n_neurons == 1

    def test_single_sample_per_group(self, tiny_probe):
        raw = _make_raw_activations(n_samples=1)
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r   = d.report("sentiment").layers["fc1"]
        assert r.n_neurons > 0

    def test_high_dimensional_tensors(self, tiny_probe):
        # (batch=2, seq=10, hidden=16) — should collapse to 16 neurons
        raw = _make_raw_activations(
            tensor_factory=lambda g, s: torch.randn(2, 10, 16)
        )
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r = d.report("sentiment").layers["fc1"]
        assert r.n_neurons == 16

    def test_multiple_layers_in_raw(self, tiny_probe):
        raw = {
            "sentiment": {
                "positive": {
                    i: {
                        "layer_a": torch.randn(1, 8),
                        "layer_b": torch.randn(1, 4),
                    }
                    for i in range(3)
                },
                "negative": {
                    i: {
                        "layer_a": torch.randn(1, 8),
                        "layer_b": torch.randn(1, 4),
                    }
                    for i in range(3)
                },
            }
        }
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        report = d.report("sentiment")
        assert "layer_a" in report.layers
        assert "layer_b" in report.layers
        assert report.layers["layer_a"].n_neurons == 8
        assert report.layers["layer_b"].n_neurons == 4

    def test_no_activations_logs_warning(self, tiny_probe, caplog):
        import logging
        empty_raw = {"sentiment": {}}   # no groups → no tensors
        d = DeadNeuronDetector(tiny_probe)
        with caplog.at_level(logging.WARNING):
            d.fit("sentiment", raw_activations=empty_raw)
        # Should not crash; report won't exist or will have empty layers
        # (either outcome is acceptable — just no exception)

    def test_dead_fraction_zero_when_all_healthy(self, tiny_probe):
        raw = _make_raw_activations(
            tensor_factory=lambda g, s: torch.ones(1, 8) * 5.0
        )
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r = d.report("sentiment")
        assert r.global_dead_fraction() == 0.0

    def test_custom_dead_threshold_changes_classification(self, tiny_probe):
        # Activation = 0.5e-5 → dead if threshold=1e-4, alive if threshold=1e-6
        val = 0.5e-5
        raw = _make_raw_activations(
            tensor_factory=lambda g, s: torch.full((1, 4), val)
        )
        d_strict = DeadNeuronDetector(tiny_probe, dead_threshold=1e-6)
        d_loose  = DeadNeuronDetector(tiny_probe, dead_threshold=1e-4)

        d_strict.fit("sentiment", raw_activations=raw)
        d_loose.fit("sentiment",  raw_activations=raw)

        r_strict = d_strict.report("sentiment").layers["fc1"]
        r_loose  = d_loose.report("sentiment").layers["fc1"]

        assert r_strict.n_dead == 0    # 5e-6 > 1e-6, so not dead
        assert r_loose.n_dead  > 0     # 5e-6 < 1e-4, so dead


class TestFitAll:

    def test_fit_all_returns_self(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        ret = d.fit_all()
        assert ret is d

    def test_fit_all_covers_every_axis(self, tiny_probe):
        tiny_probe.add_axis("domain", {
            "news":   ["breaking news today", "headlines"],
            "sports": ["final score", "championship"],
        })
        d = DeadNeuronDetector(tiny_probe)
        d.fit_all()
        axis_names = {ax.name for ax in tiny_probe._axes}
        assert axis_names == set(d._reports.keys())

    def test_fit_all_with_raw_activations(self, tiny_probe):
        # Pass multi-axis raw activations
        raw = {
            "sentiment": {
                "positive": {0: {"fc1": torch.randn(1, 8)}},
                "negative": {0: {"fc1": torch.randn(1, 8)}},
            }
        }
        d = DeadNeuronDetector(tiny_probe)
        d.fit_all(raw_activations=raw)
        # "sentiment" should use raw_activations; any other axis runs forward passes
        assert "sentiment" in d._reports


class TestReportErrorHandling:

    def test_report_wrong_axis_raises(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=_make_raw_activations())
        with pytest.raises(ValueError, match="not yet analysed"):
            d.report("nonexistent")

    def test_report_before_any_fit_raises(self, tiny_probe):
        d = DeadNeuronDetector(tiny_probe)
        with pytest.raises(ValueError):
            d.report("sentiment")


class TestSaveIntegration:

    @pytest.fixture()
    def saved_zip_with_dead(self, tiny_probe, tmp_path):
        """Run probe, fit detector, save zip with dead_neuron_reports."""
        raw = _make_raw_activations(
            groups=("positive", "negative"),
            n_samples=3,
            layer_name="fc1",
        )
        pr = _make_probe_results_with_raw(raw)

        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)

        out = pr.save(str(tmp_path / "results.zip"), dead_neuron_reports=d)
        return out, d

    def test_zip_is_valid(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        assert zipfile.is_zipfile(path)

    def test_dead_neuron_json_present(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        dead_files = [n for n in names if n.startswith("dead_neurons/")]
        assert len(dead_files) >= 1

    def test_dead_neuron_json_is_valid(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            dead_files = [n for n in zf.namelist() if n.startswith("dead_neurons/")]
            content = json.loads(zf.read(dead_files[0]))
        assert "axis_name" in content
        assert "global_summary" in content
        assert "layers" in content

    def test_neuron_raw_csvs_present(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        assert len(csv_files) >= 1

    def test_neuron_raw_csv_has_correct_columns(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            first_csv = zf.read(csv_files[0]).decode("utf-8")
        header = first_csv.splitlines()[0]
        assert "group" in header
        assert "sample_idx" in header
        assert "neuron_idx" in header
        assert "activation" in header

    def test_neuron_raw_csv_has_data_rows(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            first_csv = zf.read(csv_files[0]).decode("utf-8")
        rows = first_csv.strip().splitlines()
        assert len(rows) > 1   # at least header + 1 data row

    def test_csv_activation_values_are_floats(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            lines = zf.read(csv_files[0]).decode("utf-8").splitlines()
        # Skip header, parse first data row
        parts = lines[1].split(",")
        float(parts[-1])   # last column = activation; must be parseable as float

    def test_save_with_dead_report_object_directly(self, tiny_probe, tmp_path):
        """Pass a single DeadNeuronReport instead of detector."""
        raw = _make_raw_activations()
        pr  = _make_probe_results_with_raw(raw)
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        report = d.report("sentiment")

        out = pr.save(str(tmp_path / "direct_report.zip"), dead_neuron_reports=report)
        with zipfile.ZipFile(out) as zf:
            dead_files = [n for n in zf.namelist() if n.startswith("dead_neurons/")]
        assert len(dead_files) == 1

    def test_save_with_dict_of_reports(self, tiny_probe, tmp_path):
        """Pass a plain dict {axis_name: DeadNeuronReport}."""
        raw = _make_raw_activations()
        pr  = _make_probe_results_with_raw(raw)
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)

        out = pr.save(
            str(tmp_path / "dict_report.zip"),
            dead_neuron_reports={"sentiment": d.report("sentiment")},
        )
        assert zipfile.is_zipfile(out)

    def test_save_without_dead_reports_still_works(self, tmp_path):
        """Omitting dead_neuron_reports must not break the zip."""
        raw = _make_raw_activations()
        pr  = _make_probe_results_with_raw(raw)
        out = pr.save(str(tmp_path / "no_dead.zip"))
        assert zipfile.is_zipfile(out)
        with zipfile.ZipFile(out) as zf:
            dead_files = [n for n in zf.namelist() if n.startswith("dead_neurons/")]
        assert len(dead_files) == 0

    def test_metadata_reflects_dead_analysis(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("metadata.json"))
        assert meta["includes_dead_neuron_analysis"] is True

    def test_metadata_reflects_neuron_raw(self, saved_zip_with_dead):
        path, _ = saved_zip_with_dead
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("metadata.json"))
        assert meta["includes_neuron_raw_csv"] is True


class TestShowSmoke:

    def test_show_does_not_crash(self, tiny_probe, capsys):
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        d.report("sentiment").show()   # must not raise

    def test_show_with_neurons_does_not_crash(self, tiny_probe, capsys):
        raw = _make_raw_activations(
            tensor_factory=lambda g, s: torch.zeros(1, 6)  # all dead
        )
        d = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        d.report("sentiment").show(show_neurons=True)

    def test_show_empty_report_does_not_crash(self, tiny_probe, capsys):
        report = DeadNeuronReport(axis_name="x", model_name="m")
        report.show()   # no layers → should print "No dead neurons found"

    def test_to_dict_after_fit_is_stable(self, tiny_probe):
        """to_dict() called twice must return identical JSON."""
        raw = _make_raw_activations()
        d   = DeadNeuronDetector(tiny_probe)
        d.fit("sentiment", raw_activations=raw)
        r   = d.report("sentiment")
        assert json.dumps(r.to_dict()) == json.dumps(r.to_dict())