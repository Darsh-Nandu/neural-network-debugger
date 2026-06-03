"""
NNDbg — Deep semantic activation analysis for neural networks.

Quick start:
    from nndbg import ModelProbe

    probe = ModelProbe.from_pretrained("google/mt5-small")
    probe.add_axis("language", {
        "english": ["The cat sat on the mat.", ...],
        "french":  ["Le chat était assis.", ...],
    })
    results = probe.run()
    results.show()
"""

from nndbg.probe import ModelProbe
from nndbg.results import ProbeResults

__version__ = "0.1.0"
__all__ = ["ModelProbe", "ProbeResults"]