import torch


def test_causal_trace_recovery_matrix_shape(transformer_inspector):
    clean = torch.randint(0, 100, (1, 6))
    corrupted = clean.clone()
    corrupted[0, 2] = (corrupted[0, 2] + 1) % 100

    layers = transformer_inspector.find_layers(r"h\.\d+$")
    result = transformer_inspector.patching.causal_trace(clean, corrupted, layers=layers)
    assert result.matrix.shape == (len(layers), 6)
    assert result.layers == layers


def test_causal_trace_last_position_of_last_layer_fully_recovers(transformer_inspector):
    # The target logit only reads the LAST position's hidden state through
    # ln_f/lm_head (both position-wise). Patching the final block's output
    # at the last position with the clean run's value should therefore
    # reproduce the clean logit almost exactly -> recovery ~1 at that cell.
    clean = torch.randint(0, 100, (1, 6))
    corrupted = clean.clone()
    corrupted[0, 2] = (corrupted[0, 2] + 1) % 100

    layers = transformer_inspector.find_layers(r"h\.\d+$")
    last_layer = layers[-1]
    result = transformer_inspector.patching.causal_trace(
        clean, corrupted, layers=[last_layer], positions=list(range(6))
    )
    assert result.matrix[0, -1] > 0.9


def test_causal_trace_rejects_mismatched_lengths(transformer_inspector):
    import pytest

    clean = torch.randint(0, 100, (1, 6))
    corrupted = torch.randint(0, 100, (1, 5))
    with pytest.raises(ValueError):
        transformer_inspector.patching.causal_trace(clean, corrupted, layers=["transformer.h.0"])


def test_causal_trace_plot_runs(transformer_inspector):
    clean = torch.randint(0, 100, (1, 6))
    corrupted = clean.clone()
    corrupted[0, 1] = (corrupted[0, 1] + 1) % 100
    result = transformer_inspector.patching.causal_trace(clean, corrupted, layers=["transformer.h.0"])
    assert result.plot() is not None


def test_causal_trace_method_field(transformer_inspector):
    clean = torch.randint(0, 100, (1, 6))
    corrupted = clean.clone()
    corrupted[0, 1] = (corrupted[0, 1] + 1) % 100
    result = transformer_inspector.patching.causal_trace(clean, corrupted, layers=["transformer.h.0"])
    assert result.method == "causal_trace"


def test_mean_ablation_shape(transformer_inspector, token_dataset):
    clean = torch.randint(0, 100, (1, 6))
    layers = transformer_inspector.find_layers(r"h\.\d+$")
    result = transformer_inspector.patching.mean_ablation(clean, token_dataset, layers=layers)
    assert result.matrix.shape == (len(layers), 6)
    assert result.method == "mean_ablation"


def test_mean_ablation_plot_uses_diverging(transformer_inspector, token_dataset):
    clean = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.patching.mean_ablation(
        clean, token_dataset, layers=["transformer.h.0"]
    )
    ax = result.plot()
    assert ax is not None
