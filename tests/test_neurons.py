"""Tests for the neuron analysis plane."""
from __future__ import annotations

import torch

from nndbg.analysis.neurons.results import NeuronResult


def test_stats_shape(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    assert isinstance(result, NeuronResult)
    n = 16  # fc1 has 16 output units
    assert result.mean_activation.shape == (n,)
    assert result.max_activation.shape == (n,)
    assert result.std_activation.shape == (n,)
    assert result.kurtosis.shape == (n,)
    assert result.dead_mask.shape == (n,)
    assert result.dead_mask.dtype == torch.bool


def test_stats_dead_count_type(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    assert isinstance(result.dead_count, int)
    assert 0 <= result.dead_count <= result.n_neurons


def test_stats_top_examples_populated(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer, top_k=5)
    assert len(result.top_examples) == result.n_neurons
    for idx_list in result.top_examples.values():
        assert len(idx_list) <= 5
        assert all(0 <= i < len(inputs) for i in idx_list)


def test_top_activating_single_neuron(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.top_activating(inputs, layer=layer, neuron=0, k=3)
    assert 0 in result.top_examples
    assert len(result.top_examples) == 1
    assert len(result.top_examples[0]) <= 3


def test_polysemantic_neurons_returns_list(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    poly = result.polysemantic_neurons(percentile=25.0)
    assert isinstance(poly, list)
    assert all(isinstance(i, int) for i in poly)


def test_stats_plot_returns_axes(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    ax = result.plot()
    assert ax is not None


def test_kurtosis_plot_returns_axes(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    ax = result.plot_kurtosis()
    assert ax is not None


def test_stats_repr(mlp_inspector, mlp_probe_dataset):
    inputs = [x for x, _ in mlp_probe_dataset]
    layer = mlp_inspector.find_layers(r"fc1")[0]
    result = mlp_inspector.neurons.stats(inputs, layer=layer)
    r = repr(result)
    assert "NeuronResult" in r
    assert "dead" in r
