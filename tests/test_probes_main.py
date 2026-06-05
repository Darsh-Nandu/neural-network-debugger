"""
test_nndbg.py — Full test suite for nndbg (modified version)

Covers:
  1.  Verbose / silent mode
  2.  ProbeTrainer — all 6 probe types, feature selection, validation, repr
  3.  ProbeTrainer._build_features — correct feature subset, NaN sanitisation
  4.  ProbeTrainer.train_on_layer — CV path, split-fallback path
  5.  ProbeTrainer.train_all_layers — multi-layer dict
  6.  ProbeResults.save() — zip structure, JSON contents, tensor round-trips
  7.  ProbeResults.encoding_layers() — top_k, min_score, bad axis
  8.  ProbeResults.to_dict()
  9.  ModelProbe.from_model — tiny synthetic model, no HuggingFace download
  10. ModelProbe.add_axis — chaining, Axis validation
  11. ModelProbe.run() — end-to-end silent, probe_scores shape, raw activations
  12. ModelProbe.trace() — TraceResults shape
  13. Axis dataclass — group validation

Run with:
    PYTHONPATH=. pytest tests/test_nndbg.py -v
"""

from __future__ import annotations

import io
import json
import zipfile
import tempfile
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import nndbg
from nndbg import (
    ModelProbe,
    ProbeResults,
    ProbeTrainer,
    PROBE_TYPES,
    AVAILABLE_FEATURES,
    set_verbose,
    is_verbose,
)
from nndbg.probing.axis import Axis
from nndbg.probing.trainer import PROBE_TYPES, AVAILABLE_FEATURES
from nndbg.storage.store import ActivationStore


class TinyModel(nn.Module):
    """Minimal 2-layer MLP + tokeniser-free wrapper for fast tests."""

    def __init__(self, in_dim: int = 16, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.fc2(self.relu(self.fc1(x)))


class DummyTokenizer:
    """Returns fixed-length float tensors so we don't need a real tokenizer."""

    def __init__(self, in_dim: int = 16):
        self.in_dim = in_dim

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Deterministic tensor derived from text length so groups differ
        seed = len(text) % 100
        torch.manual_seed(seed)
        return {"x": torch.randn(1, self.in_dim)}


@pytest.fixture()
def tiny_probe():
    """A ModelProbe wrapping TinyModel — no HF download needed."""
    model = TinyModel()
    tok = DummyTokenizer()
    return ModelProbe.from_model(model, tokenizer=tok, model_name="tiny")


def _make_layer_data(
    n_samples: int = 10,
    groups: tuple = ("A", "B"),
    offset: float = 1.0,
) -> Dict[str, List[Dict]]:
    """Synthetic per-group activation stats for ProbeTrainer tests."""
    data: Dict[str, List[Dict]] = {}
    for i, g in enumerate(groups):
        data[g] = [
            {
                "mean":     float(i * offset + j * 0.01),
                "std":      0.05,
                "l2_norm":  1.0 + i * 0.5,
                "sparsity": 0.1,
                "min_val":  -0.5,
                "max_val":  0.8 + i * 0.2,
            }
            for j in range(n_samples)
        ]
    return data


def _make_probe_results(
    n_raw_samples: int = 2,
    layer_names: tuple = ("fc1", "fc2"),
) -> ProbeResults:
    """Build a ProbeResults entirely in-memory for save() tests."""
    axes = [
        Axis(
            name="sentiment",
            groups={
                "positive": ["good", "great"],
                "negative": ["bad", "awful"],
            },
        )
    ]

    probe_scores = {
        "sentiment": {ln: 0.7 + 0.1 * i for i, ln in enumerate(layer_names)}
    }

    lgd: Dict = {
        "sentiment": {
            ln: _make_layer_data(n_samples=2)
            for ln in layer_names
        }
    }

    raw: Dict = {
        "sentiment": {
            "positive": {
                i: {ln: torch.randn(2, 4) for ln in layer_names}
                for i in range(n_raw_samples)
            },
            "negative": {
                i: {ln: torch.randn(2, 4) for ln in layer_names}
                for i in range(n_raw_samples)
            },
        }
    }

    store = ActivationStore(":memory:")
    return ProbeResults(
        run_id="test001",
        model_name="test-model",
        axes=axes,
        probe_scores=probe_scores,
        store=store,
        layer_group_data=lgd,
        raw_activations=raw,
    )


class TestVerboseMode:

    def test_default_is_silent(self):
        set_verbose(False)          # reset to default
        assert is_verbose() is False

    def test_set_verbose_true(self):
        set_verbose(True)
        assert is_verbose() is True
        set_verbose(False)          # restore

    def test_set_verbose_false(self):
        set_verbose(True)
        set_verbose(False)
        assert is_verbose() is False

    def test_toggle_round_trip(self):
        original = is_verbose()
        set_verbose(not original)
        assert is_verbose() is not original
        set_verbose(original)       # restore


class TestProbeTrainerConstruction:

    def test_default_probe_type(self):
        t = ProbeTrainer()
        assert t.probe_type == "logistic"

    @pytest.mark.parametrize("pt", PROBE_TYPES)
    def test_all_probe_types_instantiate(self, pt):
        t = ProbeTrainer(probe_type=pt)
        assert t.probe_type == pt

    def test_invalid_probe_type_raises(self):
        with pytest.raises(ValueError, match="Unknown probe_type"):
            ProbeTrainer(probe_type="bananas")

    def test_default_features(self):
        t = ProbeTrainer()
        assert t.features == AVAILABLE_FEATURES

    def test_custom_features_subset(self):
        subset = ["mean", "std"]
        t = ProbeTrainer(features=subset)
        assert t.features == subset

    def test_invalid_feature_raises(self):
        with pytest.raises(ValueError, match="Invalid features"):
            ProbeTrainer(features=["mean", "not_a_feature"])

    def test_empty_features_raises(self):
        with pytest.raises(ValueError, match="At least one feature"):
            ProbeTrainer(features=[])

    def test_hyperparams_stored(self):
        t = ProbeTrainer(cv_folds=5, max_iter=500, test_size=0.2)
        assert t.cv_folds == 5
        assert t.max_iter == 500
        assert t.test_size == 0.2

    def test_repr_contains_probe_type(self):
        for pt in PROBE_TYPES:
            assert pt in repr(ProbeTrainer(probe_type=pt))


class TestBuildFeatures:

    def test_output_shape(self):
        t = ProbeTrainer()
        data = _make_layer_data(n_samples=5, groups=("X", "Y"))
        X, y = t._build_features(data)
        assert X.shape == (10, len(AVAILABLE_FEATURES))
        assert y.shape == (10,)

    def test_feature_subset_shape(self):
        t = ProbeTrainer(features=["mean", "std"])
        data = _make_layer_data(n_samples=4, groups=("A", "B"))
        X, y = t._build_features(data)
        assert X.shape[1] == 2

    def test_nan_sanitised(self):
        t = ProbeTrainer()
        data = {
            "A": [{"mean": float("nan"), "std": float("inf"),
                   "l2_norm": 1.0, "sparsity": 0.0,
                   "min_val": -1.0, "max_val": 1.0}],
            "B": [{"mean": 0.5, "std": 0.1,
                   "l2_norm": 1.0, "sparsity": 0.0,
                   "min_val": -0.5, "max_val": 0.8}],
        }
        X, y = t._build_features(data)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_labels_correspond_to_groups(self):
        t = ProbeTrainer()
        data = _make_layer_data(n_samples=3, groups=("alpha", "beta"))
        _, y = t._build_features(data)
        assert set(y.tolist()) == {"alpha", "beta"}

    def test_feature_values_match_stats(self):
        """mean column should match the mean values in the input."""
        t = ProbeTrainer(features=["mean"])
        data = {
            "A": [{"mean": 0.42, "std": 0.0, "l2_norm": 0.0,
                   "sparsity": 0.0, "min_val": 0.0, "max_val": 0.0}],
            "B": [{"mean": 0.99, "std": 0.0, "l2_norm": 0.0,
                   "sparsity": 0.0, "min_val": 0.0, "max_val": 0.0}],
        }
        X, _ = t._build_features(data)
        means = sorted(X[:, 0].tolist())
        assert pytest.approx(means, abs=1e-4) == [0.42, 0.99]


class TestProbeTrainerTraining:

    @pytest.mark.parametrize("pt", PROBE_TYPES)
    def test_all_probe_types_return_score(self, pt):
        """Every probe type must return a float in [0, 1]."""
        t = ProbeTrainer(probe_type=pt, cv_folds=3)
        data = _make_layer_data(n_samples=15, offset=2.0)  # well-separated
        score = t.train_on_layer(data)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_group_returns_zero(self):
        """Can't train a probe with only one class."""
        t = ProbeTrainer()
        data = _make_layer_data(n_samples=10, groups=("only",))
        score = t.train_on_layer(data)
        assert score == 0.0

    def test_cv_path_with_enough_samples(self):
        """With ≥ cv_folds*2 samples the CV branch should run."""
        t = ProbeTrainer(cv_folds=3)
        data = _make_layer_data(n_samples=20, offset=3.0)
        score = t.train_on_layer(data)
        assert score >= 0.0

    def test_split_fallback_few_samples(self):
        """With only 4 samples (2 per group) it falls back to train/test split."""
        t = ProbeTrainer(cv_folds=3)
        data = _make_layer_data(n_samples=2, offset=3.0)
        score = t.train_on_layer(data)
        assert 0.0 <= score <= 1.0

    def test_train_all_layers_returns_dict(self):
        layer_group_data = {
            f"layer_{i}": _make_layer_data(n_samples=10, offset=float(i + 1))
            for i in range(4)
        }
        t = ProbeTrainer()
        scores = t.train_all_layers(layer_group_data)
        assert set(scores.keys()) == set(layer_group_data.keys())
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_feature_subset_still_works(self):
        t = ProbeTrainer(features=["mean", "l2_norm"])
        data = _make_layer_data(n_samples=12, offset=2.0)
        score = t.train_on_layer(data)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("pt", PROBE_TYPES)
    def test_well_separated_data_scores_high(self, pt):
        """Groups that are far apart in feature space should be easy to classify."""
        t = ProbeTrainer(probe_type=pt, cv_folds=3)
        data = _make_layer_data(n_samples=20, offset=10.0)   # very far apart
        score = t.train_on_layer(data)
        assert score >= 0.7, f"{pt} scored only {score:.3f} on well-separated data"

    def test_three_group_probe(self):
        """Probes should work with more than 2 groups."""
        t = ProbeTrainer()
        data = _make_layer_data(n_samples=15, groups=("en", "fr", "de"), offset=5.0)
        score = t.train_on_layer(data)
        assert 0.0 <= score <= 1.0


class TestProbeResultsSave:

    @pytest.fixture()
    def saved_zip(self, tmp_path):
        pr = _make_probe_results(n_raw_samples=3)
        out = pr.save(str(tmp_path / "results.zip"))
        return out, pr

    def test_file_is_created(self, saved_zip):
        path, _ = saved_zip
        assert Path(path).exists()

    def test_zip_is_valid(self, saved_zip):
        path, _ = saved_zip
        assert zipfile.is_zipfile(path)

    def test_required_top_level_files(self, saved_zip):
        path, _ = saved_zip
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        for required in [
            "metadata.json",
            "probe_scores.json",
            "activation_stats.json",
            "layer_group_data.json",
        ]:
            assert required in names, f"Missing {required}"

    def test_metadata_json_content(self, saved_zip):
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("metadata.json"))
        assert meta["run_id"] == pr.run_id
        assert meta["model_name"] == pr.model_name
        assert len(meta["axes"]) == len(pr.axes)
        assert meta["axes"][0]["name"] == pr.axes[0].name

    def test_probe_scores_json_content(self, saved_zip):
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            scores = json.loads(zf.read("probe_scores.json"))
        assert scores == pr.probe_scores

    def test_raw_npy_tensors_present(self, saved_zip):
        path, _ = saved_zip
        with zipfile.ZipFile(path) as zf:
            npy_files = [n for n in zf.namelist() if n.endswith(".npy")]
        assert len(npy_files) > 0, "No .npy tensor files found in zip"

    def test_npy_tensor_round_trip(self, saved_zip):
        """Load a saved .npy and verify dtype + non-empty."""
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            npy_files = [n for n in zf.namelist() if n.endswith(".npy")]
            raw = zf.read(npy_files[0])
        arr = np.load(io.BytesIO(raw))
        assert arr.dtype == np.float32
        assert arr.size > 0

    def test_tensor_shape_preserved(self, saved_zip):
        """Saved tensor shape must match original torch.Tensor shape."""
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            npy_files = sorted(n for n in zf.namelist() if n.endswith(".npy"))
            raw = zf.read(npy_files[0])
        arr = np.load(io.BytesIO(raw))
        # Our synthetic tensors are (2, 4)
        assert arr.shape == (2, 4)

    def test_activation_stats_structure(self, saved_zip):
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            stats = json.loads(zf.read("activation_stats.json"))
        for axis_name in pr.probe_scores:
            assert axis_name in stats

    def test_layer_group_data_json(self, saved_zip):
        path, pr = saved_zip
        with zipfile.ZipFile(path) as zf:
            lgd = json.loads(zf.read("layer_group_data.json"))
        # Should have the same axis keys
        for axis_name in pr.probe_scores:
            assert axis_name in lgd

    def test_save_returns_absolute_path(self, tmp_path):
        pr = _make_probe_results()
        out = pr.save(str(tmp_path / "out.zip"))
        assert Path(out).is_absolute()

    def test_save_without_raw_activations(self, tmp_path):
        """save() must not crash when raw_activations is empty."""
        pr = _make_probe_results()
        pr._raw_activations = {}
        out = pr.save(str(tmp_path / "no_raw.zip"))
        assert zipfile.is_zipfile(out)
        with zipfile.ZipFile(out) as zf:
            assert "metadata.json" in zf.namelist()
            npy_files = [n for n in zf.namelist() if n.endswith(".npy")]
            assert len(npy_files) == 0

    def test_layer_names_with_dots_are_safe(self, tmp_path):
        """Layer names like 'encoder.layer.0' must produce valid zip paths."""
        pr = _make_probe_results(layer_names=("encoder.layer.0", "encoder.layer.1"))
        out = pr.save(str(tmp_path / "dots.zip"))
        with zipfile.ZipFile(out) as zf:
            npy_files = zf.namelist()
        # dots should become underscores in paths
        for n in npy_files:
            assert "//" not in n


class TestEncodingLayers:

    @pytest.fixture()
    def pr(self):
        return _make_probe_results(layer_names=("fc1", "fc2", "fc3"))

    def test_returns_list_of_tuples(self, pr):
        result = pr.encoding_layers("sentiment")
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2

    def test_sorted_descending(self, pr):
        result = pr.encoding_layers("sentiment", top_k=None)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k(self, pr):
        result = pr.encoding_layers("sentiment", top_k=2)
        assert len(result) <= 2

    def test_top_k_none_returns_all(self, pr):
        result = pr.encoding_layers("sentiment", top_k=None)
        assert len(result) == 3

    def test_min_score_filter(self, pr):
        result = pr.encoding_layers("sentiment", top_k=None, min_score=0.8)
        for _, score in result:
            assert score >= 0.8

    def test_min_score_too_high_returns_empty(self, pr):
        result = pr.encoding_layers("sentiment", min_score=2.0)
        assert result == []

    def test_invalid_axis_raises(self, pr):
        with pytest.raises(ValueError, match="not found"):
            pr.encoding_layers("nonexistent_axis")


class TestToDict:

    def test_keys_present(self):
        pr = _make_probe_results()
        d = pr.to_dict()
        for key in ("run_id", "model_name", "axes", "probe_scores"):
            assert key in d

    def test_run_id_matches(self):
        pr = _make_probe_results()
        assert pr.to_dict()["run_id"] == pr.run_id

    def test_axes_are_names(self):
        pr = _make_probe_results()
        assert pr.to_dict()["axes"] == [ax.name for ax in pr.axes]


class TestModelProbeConstruction:

    def test_from_model_stores_name(self, tiny_probe):
        assert tiny_probe.model_name == "tiny"

    def test_add_axis_returns_self(self, tiny_probe):
        result = tiny_probe.add_axis("test", {"A": ["x"], "B": ["y"]})
        assert result is tiny_probe

    def test_add_axis_chaining(self, tiny_probe):
        (
            tiny_probe
            .add_axis("axis1", {"A": ["x"], "B": ["y"]})
            .add_axis("axis2", {"C": ["c"], "D": ["d"]})
        )
        assert len(tiny_probe._axes) == 2

    def test_repr_shows_model_name(self, tiny_probe):
        assert "tiny" in repr(tiny_probe)

    def test_run_without_axes_raises(self):
        model = TinyModel()
        probe = ModelProbe.from_model(model, tokenizer=DummyTokenizer())
        with pytest.raises(ValueError, match="No axes added"):
            probe.run()


class TestModelProbeRun:

    @pytest.fixture()
    def run_results(self):
        """Run a full pipeline with the tiny synthetic model."""
        set_verbose(False)
        model = TinyModel()
        tok = DummyTokenizer()
        probe = ModelProbe.from_model(model, tokenizer=tok, model_name="tiny-e2e")
        probe.add_axis("length", {
            "short": ["hi", "ok", "no"],
            "long": [
                "The quick brown fox jumps over the lazy dog.",
                "Neural networks are universal function approximators.",
                "Activation probing reveals internal representations.",
            ],
        })
        return probe.run()

    def test_returns_probe_results(self, run_results):
        assert isinstance(run_results, ProbeResults)

    def test_probe_scores_axis_present(self, run_results):
        assert "length" in run_results.probe_scores

    def test_probe_scores_layers_are_floats(self, run_results):
        for score in run_results.probe_scores["length"].values():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_raw_activations_populated(self, run_results):
        raw = run_results._raw_activations
        assert "length" in raw
        assert "short" in raw["length"]
        assert "long" in raw["length"]

    def test_raw_activations_are_tensors(self, run_results):
        raw = run_results._raw_activations
        for group in raw["length"].values():
            for sample in group.values():
                for tensor in sample.values():
                    assert isinstance(tensor, torch.Tensor)

    def test_run_is_silent_by_default(self, capsys):
        """No stdout/stderr output should appear with verbose=False."""
        set_verbose(False)
        model = TinyModel()
        tok = DummyTokenizer()
        probe = ModelProbe.from_model(model, tokenizer=tok)
        probe.add_axis("g", {"A": ["x", "y"], "B": ["a", "b"]})
        probe.run()
        captured = capsys.readouterr()
        assert captured.out == ""
        # stderr may contain Rich ANSI but not plain INFO lines
        assert "INFO" not in captured.err

    def test_custom_probe_type_in_run(self):
        """Custom ProbeTrainer (ridge) should work end-to-end."""
        trainer = ProbeTrainer(probe_type="ridge", cv_folds=2)
        model = TinyModel()
        tok = DummyTokenizer()
        probe = ModelProbe.from_model(
            model, tokenizer=tok, probe_trainer=trainer
        )
        probe.add_axis("g", {"A": ["x", "y", "z"], "B": ["a", "b", "c"]})
        results = probe.run()
        assert isinstance(results, ProbeResults)

    def test_run_and_save(self, run_results, tmp_path):
        """end-to-end: run → save → valid zip."""
        out = run_results.save(str(tmp_path / "e2e.zip"))
        assert zipfile.is_zipfile(out)
        with zipfile.ZipFile(out) as zf:
            assert "probe_scores.json" in zf.namelist()
            npy_files = [n for n in zf.namelist() if n.endswith(".npy")]
            assert len(npy_files) > 0

    def test_encoding_layers_after_run(self, run_results):
        ranked = run_results.encoding_layers("length", top_k=None)
        assert len(ranked) > 0


class TestTrace:

    def test_returns_trace_results(self, tiny_probe):
        from nndbg.visualization.trajectory import TraceResults
        tr = tiny_probe.trace("hello world")
        assert isinstance(tr, TraceResults)

    def test_layers_are_layer_trace(self, tiny_probe):
        from nndbg.visualization.trajectory import LayerTrace
        tr = tiny_probe.trace("hello world")
        for layer in tr.layers:
            assert isinstance(layer, LayerTrace)

    def test_activations_dict_populated(self, tiny_probe):
        tr = tiny_probe.trace("test sentence")
        assert isinstance(tr.activations, dict)
        assert len(tr.activations) > 0

    def test_most_active_returns_tuples(self, tiny_probe):
        tr = tiny_probe.trace("test")
        top = tr.most_active(top_k=3)
        assert isinstance(top, list)
        for name, norm in top:
            assert isinstance(name, str)
            assert isinstance(norm, float)

    def test_text_stored(self, tiny_probe):
        text = "unique test string"
        tr = tiny_probe.trace(text)
        assert tr.text == text

    def test_model_name_stored(self, tiny_probe):
        tr = tiny_probe.trace("x")
        assert tr.model_name == "tiny"


class TestAxis:

    def test_valid_axis(self):
        ax = Axis(name="lang", groups={"en": ["hello"], "fr": ["bonjour"]})
        assert ax.name == "lang"
        assert ax.total_samples == 2

    def test_single_group_raises(self):
        with pytest.raises(ValueError, match="at least 2 groups"):
            Axis(name="bad", groups={"only": ["x"]})

    def test_empty_group_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            Axis(name="bad", groups={"A": ["x"], "B": []})

    def test_group_names_property(self):
        ax = Axis(name="x", groups={"A": ["1"], "B": ["2"], "C": ["3"]})
        assert set(ax.group_names) == {"A", "B", "C"}

    def test_total_samples_counts_all(self):
        ax = Axis(name="x", groups={"A": ["a", "b"], "B": ["c", "d", "e"]})
        assert ax.total_samples == 5

    def test_repr_contains_name(self):
        ax = Axis(name="sentiment", groups={"pos": ["x"], "neg": ["y"]})
        assert "sentiment" in repr(ax)


class TestPublicConstants:

    def test_probe_types_count(self):
        assert len(PROBE_TYPES) == 6

    def test_probe_types_values(self):
        expected = {"logistic", "ridge", "svm", "mlp", "knn", "random_forest"}
        assert set(PROBE_TYPES) == expected

    def test_available_features_count(self):
        assert len(AVAILABLE_FEATURES) == 6

    def test_available_features_values(self):
        expected = {"mean", "std", "l2_norm", "sparsity", "min_val", "max_val"}
        assert set(AVAILABLE_FEATURES) == expected

    def test_exported_from_package(self):
        """Ensure public API is reachable from the top-level package."""
        from nndbg import PROBE_TYPES as PT, AVAILABLE_FEATURES as AF
        assert len(PT) == 6
        assert len(AF) == 6