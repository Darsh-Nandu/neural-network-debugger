"""AttributionAnalyzer — which inputs caused this output?

Implements gradient-based attribution natively (no captum dependency):
saliency (Simonyan et al. 2014), gradient × input, SmoothGrad (Smilkov
et al. 2017), Integrated Gradients (Sundararajan et al. 2017), and
Grad-CAM (Selvaraju et al. 2017) adapted to sequence models.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from nndbg.analysis.attribution.results import AttributionResult

if TYPE_CHECKING:
    from nndbg.inspector import Inspector

TargetFn = Callable[[torch.Tensor], torch.Tensor]


class AttributionAnalyzer:
    """Gradient-based attribution for any differentiable model.

    For token-input models (anything exposing ``get_input_embeddings()``,
    e.g. HuggingFace causal LMs) attribution is computed per input token,
    routed through ``inputs_embeds`` so the embedding lookup stays
    differentiable. For other models, gradients are taken directly with
    respect to the input tensor.

    Example::

        result = inspector.attribution.saliency(input_ids)
        result.plot()
        inspector.attribution.integrated_gradients(input_ids).plot()
        inspector.attribution.smoothgrad(input_ids).plot()
        inspector.attribution.gradient_x_input(input_ids).plot()
    """

    def __init__(self, inspector: Inspector) -> None:
        self._inspector = inspector

    # ------------------------------------------------------------------
    def saliency(
        self, inputs: torch.Tensor, *, target=None, target_fn: TargetFn | None = None
    ) -> AttributionResult:
        """Vanilla gradient saliency: ``||d(target)/d(input)||`` per position."""
        embeds, mode = self._embed(inputs)
        embeds = embeds.clone().requires_grad_(True)
        logits = self._forward(embeds, mode)
        target_value = self._select_target(logits, target, target_fn)
        (grad,) = torch.autograd.grad(target_value, embeds)
        scores = _position_scores(grad)
        return AttributionResult(method="saliency", scores=scores.detach().cpu(), tokens=self._tokens(inputs))

    def gradient_x_input(
        self, inputs: torch.Tensor, *, target=None, target_fn: TargetFn | None = None
    ) -> AttributionResult:
        """Gradient × input: element-wise product of the gradient and the
        input embedding, a sign-aware refinement of vanilla saliency that
        cancels out large gradients at near-zero embedding dimensions."""
        embeds, mode = self._embed(inputs)
        embeds_req = embeds.clone().requires_grad_(True)
        logits = self._forward(embeds_req, mode)
        target_value = self._select_target(logits, target, target_fn)
        (grad,) = torch.autograd.grad(target_value, embeds_req)
        gxi = grad * embeds.detach()
        scores = _position_scores(gxi)
        return AttributionResult(
            method="gradient_x_input", scores=scores.detach().cpu(), tokens=self._tokens(inputs)
        )

    def smoothgrad(
        self,
        inputs: torch.Tensor,
        *,
        target=None,
        target_fn: TargetFn | None = None,
        n_samples: int = 50,
        noise_std: float = 0.1,
    ) -> AttributionResult:
        """SmoothGrad: average gradient magnitude over ``n_samples`` forward
        passes with Gaussian noise added to the input embeddings.

        Reduces the visual noise of vanilla saliency by marginalising over
        small input perturbations. Larger ``noise_std`` = more smoothing."""
        embeds, mode = self._embed(inputs)
        total_grad = torch.zeros_like(embeds)
        for _ in range(n_samples):
            noisy = (embeds + noise_std * torch.randn_like(embeds)).clone().requires_grad_(True)
            logits = self._forward(noisy, mode)
            target_value = self._select_target(logits, target, target_fn)
            (grad,) = torch.autograd.grad(target_value, noisy)
            total_grad = total_grad + grad.detach()
        scores = _position_scores(total_grad / n_samples)
        return AttributionResult(method="smoothgrad", scores=scores.cpu(), tokens=self._tokens(inputs))

    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        *,
        target=None,
        target_fn: TargetFn | None = None,
        steps: int = 50,
        baseline: torch.Tensor | None = None,
    ) -> AttributionResult:
        """Integrated Gradients: average gradient along a straight-line path
        from a baseline (zeros, by default) to the actual input, scaled by
        ``(input - baseline)``."""
        embeds, mode = self._embed(inputs)
        baseline = torch.zeros_like(embeds) if baseline is None else baseline

        total_grad = torch.zeros_like(embeds)
        for alpha in torch.linspace(0, 1, steps):
            interp = (baseline + alpha * (embeds - baseline)).clone().requires_grad_(True)
            logits = self._forward(interp, mode)
            target_value = self._select_target(logits, target, target_fn)
            (grad,) = torch.autograd.grad(target_value, interp)
            total_grad = total_grad + grad

        avg_grad = total_grad / steps
        attributions = (embeds - baseline) * avg_grad
        scores = _position_scores(attributions)
        return AttributionResult(
            method="integrated_gradients", scores=scores.detach().cpu(), tokens=self._tokens(inputs)
        )

    def gradcam(
        self,
        inputs: torch.Tensor,
        *,
        layer: str,
        target=None,
        target_fn: TargetFn | None = None,
    ) -> AttributionResult:
        """Grad-CAM: gradient-weighted activation map at ``layer``,
        collapsed across the hidden dimension into a per-position score."""
        inspector = self._inspector
        inputs = inputs.to(inspector.device)

        with inspector._hooks.capture([layer], detach=False, retain_grad=True) as cache:
            output = inspector.model(inputs)
        activation = cache[layer]
        logits = self._extract_logits(output)
        target_value = self._select_target(logits, target, target_fn)
        target_value.backward()

        grad = activation.grad
        if grad is None:
            raise RuntimeError(
                f"No gradient reached layer {layer!r} — it may not lie on the path "
                f"from the input to the selected target output."
            )
        if activation.dim() == 3:
            weights = grad.mean(dim=1, keepdim=True)  # global-avg-pool over positions
            cam = torch.relu((weights * activation).sum(dim=-1)).squeeze(0)
        else:
            cam = torch.relu(grad * activation).squeeze(0)
        return AttributionResult(
            method="gradcam", scores=cam.detach().cpu(), tokens=self._tokens(inputs), layer=layer
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _embed(self, inputs: torch.Tensor) -> tuple[torch.Tensor, str]:
        model = self._inspector.model
        inputs = inputs.to(self._inspector.device)
        if inputs.dtype in (torch.long, torch.int) and hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
            with torch.no_grad():
                embeds = embed_layer(inputs)
            return embeds, "inputs_embeds"
        return inputs.clone().float(), "inputs"

    def _forward(self, value: torch.Tensor, mode: str):
        model = self._inspector.model
        output = model(inputs_embeds=value) if mode == "inputs_embeds" else model(value)
        return self._extract_logits(output)

    @staticmethod
    def _extract_logits(output):
        return output.logits if hasattr(output, "logits") else output

    @staticmethod
    def _select_target(logits: torch.Tensor, target, target_fn: TargetFn | None):
        if target_fn is not None:
            return target_fn(logits)
        if logits.dim() == 3:  # (batch, seq, vocab) — next-token prediction
            logits = logits[:, -1, :]
        if target is None:
            target = int(logits.argmax(dim=-1)[0].item())
        return logits[:, target].sum()

    def _tokens(self, inputs: torch.Tensor) -> list[str] | None:
        tokenizer = self._inspector.tokenizer
        if tokenizer is None or inputs.dtype not in (torch.long, torch.int):
            return None
        ids = inputs[0].tolist() if inputs.dim() == 2 else inputs.tolist()
        return tokenizer.convert_ids_to_tokens(ids)


def _position_scores(grad: torch.Tensor) -> torch.Tensor:
    """Collapse a gradient/attribution tensor to one score per position.

    ``(batch, seq, hidden, ...)`` -> ``(seq,)`` via an L2 norm over every
    trailing dim. ``(batch, features)`` -> ``(features,)`` via abs value.
    """
    if grad.dim() >= 3:
        return grad.flatten(2).norm(dim=-1).squeeze(0)
    return grad.abs().squeeze(0)
