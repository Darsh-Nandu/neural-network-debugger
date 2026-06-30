"""AttentionAnalyzer — what does each head attend to?"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nndbg.analysis.attention.results import AttentionResult

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class AttentionAnalyzer:
    """Inspects attention weights of HuggingFace transformer models via
    ``output_attentions=True`` — the architecture-agnostic way to get
    attention matrices without hand-hooking each model family's internals.

    Example::

        result = inspector.attention.heads(input_ids, layer=0)
        result.plot()
        inspector.attention.rollout(input_ids).plot()
    """

    def __init__(self, inspector: Inspector) -> None:
        self._inspector = inspector

    def heads(self, input_ids: torch.Tensor, *, layer: int = -1) -> AttentionResult:
        """Attention weights for every head at one layer.

        Args:
            layer: layer index (supports negative indexing; default is the
                last layer).
        """
        attentions = self._attentions(input_ids)
        layer_idx = layer if layer >= 0 else len(attentions) + layer
        matrix = attentions[layer_idx][0]  # (heads, seq, seq)
        return AttentionResult(
            kind="heads", matrix=matrix.detach().cpu(), tokens=self._tokens(input_ids), layer=layer_idx
        )

    def rollout(self, input_ids: torch.Tensor) -> AttentionResult:
        """Attention rollout (Abnar & Zuidema, 2020): recursively multiplies
        each layer's (head-averaged, residual-augmented) attention matrix to
        approximate how information from each input token flows to every
        other token through the whole network."""
        attentions = self._attentions(input_ids)
        seq_len = attentions[0].shape[-1]
        eye = torch.eye(seq_len, device=attentions[0].device)

        flow = eye
        for layer_attn in attentions:
            avg_heads = layer_attn[0].mean(dim=0)  # (seq, seq), averaged over heads
            avg_heads = avg_heads + eye  # account for the residual connection
            avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True)
            flow = avg_heads @ flow

        return AttentionResult(kind="rollout", matrix=flow.detach().cpu(), tokens=self._tokens(input_ids))

    def entropy(self, input_ids: torch.Tensor) -> AttentionResult:
        """Per-head, per-layer entropy of the attention distribution —
        low entropy means a head focuses sharply on a few tokens, high
        entropy means it spreads attention broadly (a common proxy for
        head importance/redundancy)."""
        attentions = self._attentions(input_ids)
        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        scores = torch.zeros(n_layers, n_heads)

        for layer_idx, layer_attn in enumerate(attentions):
            probs = layer_attn[0]  # (heads, seq, seq)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=-1)
            scores[layer_idx] = ent.detach().cpu()

        return AttentionResult(kind="entropy", matrix=scores, head_scores=scores, tokens=self._tokens(input_ids))

    # ------------------------------------------------------------------
    def _attentions(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inspector = self._inspector
        input_ids = input_ids.to(inspector.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        self._force_eager_attention()

        try:
            with torch.no_grad():
                output = inspector.model(input_ids, output_attentions=True)
        except TypeError as exc:
            raise RuntimeError(
                f"{type(inspector.model).__name__} does not accept output_attentions=True. "
                f"inspector.attention requires a HuggingFace transformer model."
            ) from exc
        attentions = getattr(output, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "Model did not return attentions. inspector.attention requires a "
                "HuggingFace model that supports output_attentions=True (most "
                "encoder/decoder/causal-LM models do)."
            )
        return attentions

    def _force_eager_attention(self) -> None:
        """SDPA/flash-attention kernels don't materialize attention weights,
        so output_attentions=True silently returns None under them. Switch
        the model to the 'eager' implementation, which always does."""
        config = getattr(self._inspector.model, "config", None)
        if config is not None and getattr(config, "_attn_implementation", None) != "eager":
            config._attn_implementation = "eager"

    def _tokens(self, input_ids: torch.Tensor) -> list[str] | None:
        tokenizer = self._inspector.tokenizer
        if tokenizer is None:
            return None
        ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
        return tokenizer.convert_ids_to_tokens(ids)
