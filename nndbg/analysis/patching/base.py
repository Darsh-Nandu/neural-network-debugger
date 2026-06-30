"""PatchingAnalyzer — which layers causally produce a behaviour?

Implements activation patching / causal tracing in the style of Meng et al.
(2022), "Locating and Editing Factual Associations in GPT" (ROME): run a
clean and a corrupted input through the model, then for every (layer,
position) pair, splice the clean run's activation into the corrupted run
and measure how much of the target logit gap that single substitution
recovers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nndbg.analysis.patching.results import PatchingResult

if TYPE_CHECKING:
    from nndbg.inspector import Inspector


class PatchingAnalyzer:
    """Example::

        clean = tokenizer("The Eiffel Tower is in the city of", return_tensors="pt").input_ids
        corrupted = tokenizer("The Space Needle is in the city of", return_tensors="pt").input_ids
        target = tokenizer(" Paris").input_ids[0]

        result = inspector.patching.causal_trace(
            clean, corrupted, target=target, layers=inspector.find_layers(r"h\\.\\d+$")
        )
        result.plot()
    """

    def __init__(self, inspector: "Inspector") -> None:
        self._inspector = inspector

    def causal_trace(
        self,
        clean: torch.Tensor,
        corrupted: torch.Tensor,
        *,
        target: int | None = None,
        layers: list[str] | None = None,
        positions: list[int] | None = None,
    ) -> PatchingResult:
        """Args:
            clean: the input that produces the desired behaviour, shape
                ``(1, seq)``.
            corrupted: a same-length input that produces the wrong
                behaviour (e.g. a key entity swapped out).
            target: vocabulary index to track. Defaults to the clean run's
                argmax prediction at the last position.
            layers: layer names to scan. Defaults to every registered
                layer — for transformer models, restrict this to one
                module per block (e.g. ``inspector.find_layers(r"h\\.\\d+$")``)
                to keep the ``len(layers) * len(positions)`` forward passes
                tractable.
            positions: token positions to scan. Defaults to every position.
        """
        inspector = self._inspector
        clean = self._prep(clean)
        corrupted = self._prep(corrupted)
        if clean.shape != corrupted.shape:
            raise ValueError(
                f"clean and corrupted must have the same shape for causal tracing, "
                f"got {tuple(clean.shape)} and {tuple(corrupted.shape)}."
            )

        layer_names = layers or inspector.layers()
        seq_len = clean.shape[-1]
        position_list = positions if positions is not None else list(range(seq_len))

        with inspector._hooks.capture(layer_names) as clean_cache, torch.no_grad():
            clean_logits = self._extract_logits(inspector.model(clean))
        with inspector._hooks.capture(layer_names) as corrupted_cache, torch.no_grad():
            corrupted_logits = self._extract_logits(inspector.model(corrupted))

        target_idx = self._resolve_target(clean_logits, target)
        clean_value = self._target_value(clean_logits, target_idx)
        corrupted_value = self._target_value(corrupted_logits, target_idx)
        denom = clean_value - corrupted_value

        recovery = torch.zeros(len(layer_names), len(position_list))
        for i, layer in enumerate(layer_names):
            clean_act = clean_cache[layer]
            corrupted_act = corrupted_cache[layer]
            for j, pos in enumerate(position_list):
                patched_act = corrupted_act.clone()
                patched_act[:, pos, ...] = clean_act[:, pos, ...]
                with inspector._hooks.patch({layer: patched_act}), torch.no_grad():
                    patched_logits = self._extract_logits(inspector.model(corrupted))
                patched_value = self._target_value(patched_logits, target_idx)
                recovery[i, j] = (
                    (patched_value - corrupted_value) / denom if denom.abs() > 1e-8 else torch.tensor(0.0)
                )

        return PatchingResult(
            matrix=recovery,
            layers=layer_names,
            positions=position_list,
            tokens=self._tokens(corrupted),
            target=target_idx,
        )

    # ------------------------------------------------------------------
    def _prep(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self._inspector.device)
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    @staticmethod
    def _extract_logits(output: torch.Tensor):
        return output.logits if hasattr(output, "logits") else output

    @staticmethod
    def _resolve_target(clean_logits: torch.Tensor, target: int | None) -> int:
        if target is not None:
            return int(target)
        logits = clean_logits[:, -1, :] if clean_logits.dim() == 3 else clean_logits
        return int(logits.argmax(dim=-1)[0].item())

    @staticmethod
    def _target_value(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
        logits = logits[:, -1, :] if logits.dim() == 3 else logits
        return logits[0, target_idx]

    def _tokens(self, input_ids: torch.Tensor) -> list[str] | None:
        tokenizer = self._inspector.tokenizer
        if tokenizer is None:
            return None
        return tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
