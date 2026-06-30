"""Shared color palette and styling constants for NNDbg plots."""
from __future__ import annotations

# Qualitative palette for categorical series (probe groups, SAE features, ...)
PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
]

# Sequential colormap for heatmaps (attention, attribution, patching)
HEATMAP_CMAP = "viridis"

# Diverging colormap for signed values (saliency, causal-tracing recovery delta)
DIVERGING_CMAP = "RdBu_r"

FIGSIZE = (7, 4.5)
DPI = 110
FONT_SIZE = 10


def apply_style(ax) -> None:
    """Apply consistent, minimal styling to a matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.xaxis.label.set_size(FONT_SIZE)
    ax.yaxis.label.set_size(FONT_SIZE)
    ax.title.set_size(FONT_SIZE + 1)
