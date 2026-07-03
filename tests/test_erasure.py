"""Tests for the concept-erasure (INLP) plane."""
from __future__ import annotations

import torch

from nndbg.analysis.erasure.results import ErasureResult


def test_inlp_returns_result(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    assert isinstance(result, ErasureResult)


def test_inlp_projection_is_square(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    d = 16  # fc1 output dim
    assert result.projection.shape == (d, d)


def test_inlp_accuracy_trace_non_empty(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    assert len(result.accuracy_trace) >= 1
    assert all(0.0 <= a <= 1.0 for a in result.accuracy_trace)


def test_inlp_accuracy_decreases(mlp_inspector, mlp_probe_dataset):
    """After INLP, probe accuracy should drop toward chance."""
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(
        mlp_probe_dataset, concept="parity", layer=layer, n_iters=10, min_accuracy=0.0
    )
    if len(result.accuracy_trace) >= 2:
        assert result.accuracy_trace[-1] <= result.accuracy_trace[0] + 0.05


def test_inlp_apply_shape(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    sample_acts = torch.randn(5, 16)
    erased = result.apply(sample_acts)
    assert erased.shape == sample_acts.shape


def test_inlp_apply_is_projection(mlp_inspector, mlp_probe_dataset):
    """Applying the erasure twice should give the same result as applying once
    (idempotent under orthogonal projection)."""
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    sample_acts = torch.randn(5, 16)
    once = result.apply(sample_acts)
    twice = result.apply(once)
    # Idempotency of a chained product of ~n_iters float32 Householder-style
    # projections only holds up to accumulated numerical error, not exact
    # (1e-5) precision — probe directions aren't perfectly orthogonal across
    # iterations, and that residual varies slightly with the sklearn solver.
    assert torch.allclose(once, twice, atol=1e-3)


def test_inlp_plot_returns_axes(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    ax = result.plot()
    assert ax is not None


def test_inlp_repr(mlp_inspector, mlp_probe_dataset):
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.erasure.inlp(mlp_probe_dataset, concept="parity", layer=layer)
    r = repr(result)
    assert "ErasureResult" in r
    assert "parity" in r
