def test_vae_training_loss_decreases(transformer_inspector, token_dataset):
    vae = transformer_inspector.latent.train(token_dataset, layer="transformer.h.0", latent_dim=2, epochs=30)
    assert vae.loss_history[-1] < vae.loss_history[0]


def test_vae_encode_shapes(transformer_inspector, token_dataset):
    transformer_inspector.latent.train(token_dataset, layer="transformer.h.0", latent_dim=2, epochs=10)
    result = transformer_inspector.latent.encode(token_dataset, layer="transformer.h.0")
    assert result.latent.shape == (len(token_dataset), 2)
    assert result.reconstruction_error.shape == (len(token_dataset),)


def test_vae_anomalies_returns_valid_indices(transformer_inspector, token_dataset):
    transformer_inspector.latent.train(token_dataset, layer="transformer.h.0", latent_dim=2, epochs=10)
    result = transformer_inspector.latent.encode(token_dataset, layer="transformer.h.0")
    anomalies = result.anomalies()
    assert all(0 <= i < len(token_dataset) for i in anomalies)


def test_vae_encode_without_training_raises(transformer_inspector, token_dataset):
    import pytest

    with pytest.raises(RuntimeError):
        transformer_inspector.latent.encode(token_dataset, layer="transformer.h.1")


def test_vae_higher_dim_latent_projects_to_2d_for_plotting(transformer_inspector, token_dataset):
    transformer_inspector.latent.train(token_dataset, layer="transformer.h.0", latent_dim=4, epochs=10)
    result = transformer_inspector.latent.encode(token_dataset, layer="transformer.h.0")
    assert result.latent.shape[1] == 4
    assert result.plot() is not None  # exercises the PCA-projection path


def test_vae_plots_run(transformer_inspector, token_dataset):
    transformer_inspector.latent.train(token_dataset, layer="transformer.h.0", latent_dim=2, epochs=10)
    result = transformer_inspector.latent.encode(token_dataset, layer="transformer.h.0")
    result.plot()
    result.plot_training()
