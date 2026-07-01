def test_sae_training_loss_decreases(transformer_inspector, token_dataset):
    sae = transformer_inspector.sae.train(token_dataset, layer="transformer.h.0", n_features=32, epochs=15)
    assert sae.loss_history[-1] < sae.loss_history[0]


def test_sae_decompose_shapes(transformer_inspector, token_dataset):
    transformer_inspector.sae.train(token_dataset, layer="transformer.h.0", n_features=32, epochs=5)
    result = transformer_inspector.sae.decompose(token_dataset, layer="transformer.h.0")
    assert result.feature_activations.shape == (len(token_dataset), 32)
    assert 0.0 <= result.sparsity <= 1.0
    assert result.reconstruction_loss >= 0.0


def test_sae_top_features_and_repr(transformer_inspector, token_dataset):
    transformer_inspector.sae.train(token_dataset, layer="transformer.h.0", n_features=32, epochs=5)
    result = transformer_inspector.sae.decompose(token_dataset, layer="transformer.h.0")
    top = result.top_features(5)
    assert len(top) == 5
    assert "transformer.h.0" in repr(result)


def test_sae_decompose_without_training_raises(transformer_inspector, token_dataset):
    import pytest

    with pytest.raises(RuntimeError):
        transformer_inspector.sae.decompose(token_dataset, layer="transformer.h.1")


def test_sae_plots_run(transformer_inspector, token_dataset):
    transformer_inspector.sae.train(token_dataset, layer="transformer.h.0", n_features=16, epochs=5)
    result = transformer_inspector.sae.decompose(token_dataset, layer="transformer.h.0")
    result.plot()
    result.plot_training()


def test_sae_topk_activation_exact_sparsity(transformer_inspector, token_dataset):
    k = 4
    transformer_inspector.sae.train(
        token_dataset, layer="transformer.h.1", n_features=32, epochs=5, activation=f"topk:{k}"
    )
    result = transformer_inspector.sae.decompose(token_dataset, layer="transformer.h.1")
    # Each example has exactly k active features; all others should be 0
    active_per_example = (result.feature_activations.abs() > 1e-6).sum(dim=-1)
    assert (active_per_example <= k).all()


def test_sae_topk_invalid_raises():
    import pytest

    from nndbg.analysis.sae.base import SparseAutoencoder

    with pytest.raises(ValueError, match="activation"):
        SparseAutoencoder(16, 64, activation="invalid")
