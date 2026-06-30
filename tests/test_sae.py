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
