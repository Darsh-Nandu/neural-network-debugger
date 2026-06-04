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
    results.save("my_run.zip")

Verbose mode (logs + progress bars):
    import nndbg
    nndbg.set_verbose(True)

Custom probe type:
    from nndbg import ModelProbe, ProbeTrainer

    trainer = ProbeTrainer(probe_type="svm", features=["mean", "std", "l2_norm"])
    probe   = ModelProbe.from_pretrained("google/mt5-small", probe_trainer=trainer)
"""

from nndbg.probe import ModelProbe
from nndbg.results import ProbeResults
from nndbg.probing.trainer import ProbeTrainer, PROBE_TYPES, AVAILABLE_FEATURES
from nndbg.utils.logging import set_verbose, is_verbose

__version__ = "0.1.0"
__all__ = [
    "ModelProbe",
    "ProbeResults",
    "ProbeTrainer",
    "PROBE_TYPES",
    "AVAILABLE_FEATURES",
    "set_verbose",
    "is_verbose",
]