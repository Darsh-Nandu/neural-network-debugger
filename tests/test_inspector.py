from nndbg import Inspector


def test_inspector_lazy_analyzers_are_singletons(mlp_inspector):
    probing_a = mlp_inspector.probing
    probing_b = mlp_inspector.probing
    assert probing_a is probing_b


def test_inspector_layers_and_find_layers(mlp_inspector):
    assert set(mlp_inspector.layers()) == {"fc1", "act", "fc2"}
    assert mlp_inspector.find_layers(r"^fc") == ["fc1", "fc2"]


def test_inspector_repr(mlp_inspector):
    text = repr(mlp_inspector)
    assert "TinyMLP" in text
    assert "layers=3" in text


def test_inspector_summary_runs(mlp_inspector, capsys):
    mlp_inspector.summary()
    captured = capsys.readouterr()
    assert "TinyMLP" in captured.out


def test_inspector_is_exported_from_top_level_package():
    assert Inspector is not None
