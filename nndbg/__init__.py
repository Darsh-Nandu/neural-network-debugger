"""
NNDbg — A diagnostic toolkit for neural networks.

Built around a single entrypoint, ``Inspector``, exposing focused analysis
planes (probing, attribution, attention, activation patching, sparse
autoencoders, VAE latent analysis) for PyTorch and HuggingFace models.
"""

from nndbg._version import __version__
from nndbg.utils.logging import set_verbose, is_verbose

__all__ = ["__version__", "set_verbose", "is_verbose"]
