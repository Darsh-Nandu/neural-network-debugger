import numpy as np

from nndbg.viz import plotting


def test_heatmap_returns_axes():
    ax = plotting.heatmap(np.random.randn(4, 4), xticklabels=list("abcd"), yticklabels=list("wxyz"))
    assert ax is not None


def test_heatmap_diverging_centers_at_zero():
    ax = plotting.heatmap(np.array([[-2.0, 0.0], [1.0, 2.0]]), diverging=True)
    im = ax.images[0]
    vmin, vmax = im.get_clim()
    assert vmin == -vmax


def test_line_accepts_single_series_and_dict_series():
    ax1 = plotting.line([1, 2, 3], [0.1, 0.2, 0.3])
    ax2 = plotting.line([1, 2, 3], {"a": [0.1, 0.2, 0.3], "b": [0.3, 0.2, 0.1]})
    assert ax1 is not None and ax2 is not None


def test_scatter_with_and_without_labels():
    x, y = np.random.randn(10), np.random.randn(10)
    assert plotting.scatter(x, y) is not None
    assert plotting.scatter(x, y, labels=np.random.randint(0, 3, 10)) is not None


def test_bar_returns_axes():
    ax = plotting.bar(["a", "b", "c"], [1, 2, 3])
    assert ax is not None


def test_plotly_backend_raises_helpful_error_when_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from nndbg.viz import plotly_backend

    try:
        plotly_backend.bar(["a"], [1])
    except ImportError as exc:
        assert "nndbg[plotly]" in str(exc)
    else:
        raise AssertionError("expected ImportError")
