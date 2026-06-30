import torch


def test_heads_returns_per_head_attention_matrix(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attention.heads(input_ids, layer=0)
    assert result.matrix.shape == (2, 6, 6)  # n_head=2 in the fixture config
    assert result.layer == 0


def test_heads_negative_layer_index(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attention.heads(input_ids, layer=-1)
    assert result.layer == 1  # n_layer=2 -> last index is 1


def test_attention_rows_sum_to_one(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attention.heads(input_ids, layer=0)
    row_sums = result.matrix.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_rollout_shape_and_row_sums(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attention.rollout(input_ids)
    assert result.matrix.shape == (6, 6)
    row_sums = result.matrix.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_entropy_shape(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    result = transformer_inspector.attention.entropy(input_ids)
    assert result.head_scores.shape == (2, 2)  # (n_layer, n_head)
    assert (result.head_scores >= 0).all()


def test_attention_plots_run_without_error(transformer_inspector):
    input_ids = torch.randint(0, 100, (1, 6))
    transformer_inspector.attention.heads(input_ids, layer=0).plot()
    transformer_inspector.attention.rollout(input_ids).plot()
    transformer_inspector.attention.entropy(input_ids).plot()


def test_attention_on_non_transformer_model_raises(mlp_inspector):
    import pytest

    with pytest.raises(RuntimeError):
        mlp_inspector.attention.heads(torch.randn(1, 4))
