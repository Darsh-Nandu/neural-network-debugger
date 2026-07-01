"""Tests for the geometry analysis plane (CKA, PCA, cross-model compare)."""
from __future__ import annotations

import pytest
import torch

from nndbg.analysis.geometry.results import GeometryResult, ProjectionResult


def test_layer_similarity_shape(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layers = mlp_inspector.find_layers(r"fc\d|act")
    result = mlp_inspector.geometry.layer_similarity(inputs, layers=layers)
    assert isinstance(result, GeometryResult)
    n = len(layers)
    assert result.similarity.shape == (n, n)
    assert result.row_layers == layers
    assert result.col_layers == layers


def test_layer_similarity_diagonal_is_one(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layers = mlp_inspector.find_layers(r"fc\d")
    result = mlp_inspector.geometry.layer_similarity(inputs, layers=layers)
    diag = result.similarity.diagonal()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)


def test_layer_similarity_symmetric(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layers = mlp_inspector.find_layers(r"fc\d")
    result = mlp_inspector.geometry.layer_similarity(inputs, layers=layers)
    assert torch.allclose(result.similarity, result.similarity.T, atol=1e-5)


def test_layer_similarity_values_in_range(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    result = mlp_inspector.geometry.layer_similarity(inputs)
    assert result.similarity.min() >= -0.01
    assert result.similarity.max() <= 1.01


def test_compare_cross_model(mlp_inspector, tiny_mlp):
    import torch.nn as nn

    from nndbg.inspector import Inspector

    torch.manual_seed(99)

    class TinyMLP2(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 16)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(16, 3)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    other = Inspector(TinyMLP2())
    inputs = [torch.randn(1, 4) for _ in range(20)]
    layers_a = mlp_inspector.find_layers(r"fc\d")
    layers_b = other.find_layers(r"fc\d")
    result = mlp_inspector.geometry.compare(other, inputs, layers=layers_a, other_layers=layers_b)
    assert result.similarity.shape == (len(layers_a), len(layers_b))
    assert result.row_layers == layers_a
    assert result.col_layers == layers_b


def test_pca_shape(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.geometry.pca(inputs, layer=layer, n_components=2)
    assert isinstance(result, ProjectionResult)
    assert result.coords.shape == (len(inputs), 2)
    assert len(result.explained_variance_ratio) == 2
    assert sum(result.explained_variance_ratio) <= 1.01


def test_pca_with_labels(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    labels = [y for _, y in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.geometry.pca(inputs, layer=layer, labels=labels)
    assert result.labels == labels


def test_pca_plot_returns_axes(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.geometry.pca(inputs, layer=layer)
    ax = result.plot()
    assert ax is not None


def test_geometry_result_plot(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layers = mlp_inspector.find_layers(r"fc\d")
    result = mlp_inspector.geometry.layer_similarity(inputs, layers=layers)
    ax = result.plot()
    assert ax is not None


def test_umap_raises_importerror_without_umap_learn(mlp_inspector, mlp_probe_dataset, monkeypatch):
    """When umap-learn is absent, GeometryAnalyzer.umap() should raise ImportError."""
    import sys

    monkeypatch.setitem(sys.modules, "umap", None)  # simulate package missing
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    with pytest.raises(ImportError, match="umap-learn"):
        mlp_inspector.geometry.umap(inputs, layer=layer)
