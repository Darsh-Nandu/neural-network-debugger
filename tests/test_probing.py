import pytest
import torch


def test_probe_fit_returns_accuracy_per_layer(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1", "act", "fc2"], cv=3
    )
    assert result.layers == ["fc1", "act", "fc2"]
    assert len(result.accuracy) == 3
    for acc in result.accuracy:
        assert 0.0 <= acc <= 1.0


def test_probe_finds_a_strongly_separable_concept(mlp_inspector, mlp_probe_dataset):
    # mlp_probe_dataset encodes the label as a +5 mean shift on the input,
    # which fc1 passes through almost linearly — the probe should do well.
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1"], cv=3
    )
    assert result.accuracy[0] > 0.8


def test_probe_result_best_layer_and_repr(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1", "fc2"], cv=3
    )
    assert result.best_layer() in result.layers
    assert "shift" in repr(result)


def test_probe_plot_returns_axes(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1"], cv=3
    )
    ax = result.plot()
    assert ax is not None


def test_probe_rejects_single_class(mlp_inspector):
    dataset = [(torch.randn(1, 4), 0) for _ in range(10)]
    with pytest.raises(ValueError):
        mlp_inspector.probing.fit(dataset, concept="degenerate", layers=["fc1"])


def test_probe_method_svm(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1"], cv=3, method="svm"
    )
    assert result.method == "svm"
    assert result.accuracy[0] > 0.8


def test_probe_method_mlp(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1"], cv=3, method="mlp"
    )
    assert result.method == "mlp"
    assert 0.0 <= result.accuracy[0] <= 1.0


def test_probe_method_in_repr(mlp_inspector, mlp_probe_dataset):
    result = mlp_inspector.probing.fit(
        mlp_probe_dataset, concept="shift", layers=["fc1"], cv=3, method="svm"
    )
    assert "svm" in repr(result)


def test_probe_invalid_method_raises(mlp_inspector, mlp_probe_dataset):
    with pytest.raises(ValueError, match="method"):
        mlp_inspector.probing.fit(mlp_probe_dataset, concept="x", layers=["fc1"], method="xgboost")
