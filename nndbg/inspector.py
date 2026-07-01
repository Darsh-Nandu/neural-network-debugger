"""Inspector — the single entrypoint into NNDbg.

Every analysis plane is exposed as a lazily-initialised sub-analyzer so
that importing nndbg, or constructing an Inspector, never pulls in a
plane's extra cost (sklearn, matplotlib) until you actually use it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nndbg.core.hooks import HookManager
from nndbg.core.registry import LayerRegistry

if TYPE_CHECKING:
    from nndbg.analysis.attention.base import AttentionAnalyzer
    from nndbg.analysis.attribution.base import AttributionAnalyzer
    from nndbg.analysis.erasure.base import ErasureAnalyzer
    from nndbg.analysis.geometry.base import GeometryAnalyzer
    from nndbg.analysis.latent.base import LatentAnalyzer
    from nndbg.analysis.neurons.base import NeuronAnalyzer
    from nndbg.analysis.patching.base import PatchingAnalyzer
    from nndbg.analysis.probing.base import ProbingAnalyzer
    from nndbg.analysis.sae.base import SAEAnalyzer


class Inspector:
    """Unified interface to all NNDbg analysis planes.

    Args:
        model: any ``nn.Module`` — a raw PyTorch model or a HuggingFace
            ``PreTrainedModel``.
        tokenizer: optional HuggingFace tokenizer, used by planes that
            render token-level results (attribution, attention).
        device: torch device string/object. Defaults to CUDA if available,
            else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self._registry = LayerRegistry(model)
        self._hooks = HookManager(model, self._registry)

        self._probing: ProbingAnalyzer | None = None
        self._attribution: AttributionAnalyzer | None = None
        self._attention: AttentionAnalyzer | None = None
        self._patching: PatchingAnalyzer | None = None
        self._sae: SAEAnalyzer | None = None
        self._latent: LatentAnalyzer | None = None
        self._geometry: GeometryAnalyzer | None = None
        self._neurons: NeuronAnalyzer | None = None
        self._erasure: ErasureAnalyzer | None = None

    # ------------------------------------------------------------------
    # Sub-analyzer properties (lazy init)
    # ------------------------------------------------------------------
    @property
    def probing(self) -> ProbingAnalyzer:
        if self._probing is None:
            from nndbg.analysis.probing.base import ProbingAnalyzer

            self._probing = ProbingAnalyzer(self)
        return self._probing

    @property
    def attribution(self) -> AttributionAnalyzer:
        if self._attribution is None:
            from nndbg.analysis.attribution.base import AttributionAnalyzer

            self._attribution = AttributionAnalyzer(self)
        return self._attribution

    @property
    def attention(self) -> AttentionAnalyzer:
        if self._attention is None:
            from nndbg.analysis.attention.base import AttentionAnalyzer

            self._attention = AttentionAnalyzer(self)
        return self._attention

    @property
    def patching(self) -> PatchingAnalyzer:
        if self._patching is None:
            from nndbg.analysis.patching.base import PatchingAnalyzer

            self._patching = PatchingAnalyzer(self)
        return self._patching

    @property
    def sae(self) -> SAEAnalyzer:
        if self._sae is None:
            from nndbg.analysis.sae.base import SAEAnalyzer

            self._sae = SAEAnalyzer(self)
        return self._sae

    @property
    def latent(self) -> LatentAnalyzer:
        if self._latent is None:
            from nndbg.analysis.latent.base import LatentAnalyzer

            self._latent = LatentAnalyzer(self)
        return self._latent

    @property
    def geometry(self) -> GeometryAnalyzer:
        if self._geometry is None:
            from nndbg.analysis.geometry.base import GeometryAnalyzer

            self._geometry = GeometryAnalyzer(self)
        return self._geometry

    @property
    def neurons(self) -> NeuronAnalyzer:
        if self._neurons is None:
            from nndbg.analysis.neurons.base import NeuronAnalyzer

            self._neurons = NeuronAnalyzer(self)
        return self._neurons

    @property
    def erasure(self) -> ErasureAnalyzer:
        if self._erasure is None:
            from nndbg.analysis.erasure.base import ErasureAnalyzer

            self._erasure = ErasureAnalyzer(self)
        return self._erasure

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def layers(self) -> list[str]:
        """Return a list of all registered layer names."""
        return self._registry.names()

    def find_layers(self, pattern: str) -> list[str]:
        """Return layer names matching a regex pattern."""
        return self._registry.find(pattern)

    def summary(self) -> None:
        """Pretty-print a summary of the model and available analysis planes."""
        from nndbg.viz.summary import print_summary

        print_summary(self)

    def __repr__(self) -> str:
        return (
            f"Inspector(model={type(self.model).__name__}, "
            f"device={self.device}, layers={len(self.layers())})"
        )
