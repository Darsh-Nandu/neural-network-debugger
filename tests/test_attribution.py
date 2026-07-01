import torch


def test_saliency_shape_matches_input_features(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.saliency(x)
    assert result.scores.shape == (4,)


def test_integrated_gradients_shape_matches_input_features(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.integrated_gradients(x, steps=5)
    assert result.scores.shape == (4,)


def test_gradcam_shape_matches_layer_output(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.gradcam(x, layer="fc1")
    assert result.scores.shape == (16,)
    assert result.layer == "fc1"


def test_attribution_scores_are_non_negative_for_gradcam(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.gradcam(x, layer="fc1")
    assert (result.scores >= 0).all()


def test_attribution_plot_returns_axes(mlp_inspector):
    result = mlp_inspector.attribution.saliency(torch.randn(1, 4))
    assert result.plot() is not None


def test_saliency_on_token_inputs_returns_one_score_per_token(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attribution.saliency(input_ids)
    assert result.scores.shape == (6,)


def test_integrated_gradients_on_token_inputs(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attribution.integrated_gradients(input_ids, steps=5)
    assert result.scores.shape == (6,)


def test_gradcam_on_transformer_block(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attribution.gradcam(input_ids, layer="transformer.h.0")
    assert result.scores.shape == (6,)


def test_gradient_x_input_shape(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.gradient_x_input(x)
    assert result.scores.shape == (4,)
    assert result.method == "gradient_x_input"


def test_smoothgrad_shape(mlp_inspector):
    x = torch.randn(1, 4)
    result = mlp_inspector.attribution.smoothgrad(x, n_samples=5)
    assert result.scores.shape == (4,)
    assert result.method == "smoothgrad"


def test_smoothgrad_on_token_inputs(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attribution.smoothgrad(input_ids, n_samples=5)
    assert result.scores.shape == (6,)


def test_gradient_x_input_on_token_inputs(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attribution.gradient_x_input(input_ids)
    assert result.scores.shape == (6,)
