"""Optional interactive plotting backend — only used by ``Result.plotly()``.

Plotly is not a core dependency. Every function here imports it lazily and
raises a helpful error if it isn't installed, so importing nndbg (or even
calling ``.plot()``) never requires plotly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive plots require plotly. Install it with: pip install nndbg[plotly]"
        ) from exc
    return go


def heatmap(matrix, *, xticklabels=None, yticklabels=None, title=None, diverging=False):
    go = _require_plotly()
    matrix = np.asarray(matrix)
    colorscale = "RdBu" if diverging else "Viridis"
    zmid = 0 if diverging else None
    fig = go.Figure(
        go.Heatmap(z=matrix, x=xticklabels, y=yticklabels, colorscale=colorscale, zmid=zmid)
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def line(x: Sequence, series: dict[str, Sequence[float]], *, title=None, xlabel=None, ylabel=None):
    go = _require_plotly()
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_trace(go.Scatter(x=list(x), y=list(ys), mode="lines+markers", name=name or "value"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
    return fig


def scatter(x, y, *, labels=None, title=None, xlabel=None, ylabel=None):
    go = _require_plotly()
    fig = go.Figure(
        go.Scatter(
            x=list(x), y=list(y), mode="markers",
            marker=dict(color=labels, colorscale="Viridis", showscale=labels is not None),
            text=[str(v) for v in labels] if labels is not None else None,
        )
    )
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
    return fig


def bar(labels: Sequence[str], values: Sequence[float], *, title=None):
    go = _require_plotly()
    fig = go.Figure(go.Bar(x=list(labels), y=list(values)))
    fig.update_layout(title=title, template="plotly_white")
    return fig
