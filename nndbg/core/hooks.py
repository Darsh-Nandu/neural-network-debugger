"""HookManager — forward-hook activation capture and activation patching."""
from __future__ import annotations

import contextlib
from typing import Iterable

import torch
import torch.nn as nn

from nndbg.core.cache import ActivationCache
from nndbg.core.registry import LayerRegistry


class HookManager:
    """Attaches forward hooks to named layers of a model.

    Two modes, both scoped to a ``with`` block so hooks are always cleaned
    up even if the forward pass raises:

    * :meth:`capture` — record each layer's output into an
      :class:`ActivationCache`.
    * :meth:`patch` — replace each layer's output with a supplied tensor
      during the forward pass (used for activation patching / causal
      tracing).
    """

    def __init__(self, model: nn.Module, registry: LayerRegistry | None = None) -> None:
        self.model = model
        self.registry = registry or LayerRegistry(model)

    @contextlib.contextmanager
    def capture(
        self,
        layers: Iterable[str],
        *,
        detach: bool = True,
        retain_grad: bool = False,
        capture_input: bool = False,
    ):
        """Capture the output of each named layer during a forward pass.

        Args:
            layers: layer names (see ``registry.names()``).
            detach: detach + clone captured tensors off the autograd graph.
                Set ``False`` when a gradient-based method (attribution,
                GradCAM) needs to backward through the captured activation.
            retain_grad: call ``.retain_grad()`` on captured tensors so
                ``cache[name].grad`` is populated after ``loss.backward()``.
                Only meaningful when ``detach=False``.
            capture_input: also capture the module's input, stored under
                ``f"{layer}::input"``.

        Yields:
            An :class:`ActivationCache` populated once the block exits the
            forward pass (it's filled live, so it can also be read inside
            the ``with`` block right after the forward call).
        """
        cache = ActivationCache()
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for name in layers:
            module = self.registry[name]
            handles.append(
                module.register_forward_hook(
                    _capture_hook(name, cache, detach, retain_grad, capture_input)
                )
            )

        try:
            yield cache
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def patch(self, replacements: dict[str, torch.Tensor]):
        """Replace named layers' outputs during the forward pass.

        Args:
            replacements: mapping of layer name -> tensor to substitute in
                place of that layer's normal output. The tensor must match
                the shape the model expects at that point (build it by
                cloning a captured activation and overwriting the slice you
                want to patch).
        """
        handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, value in replacements.items():
            module = self.registry[name]
            handles.append(module.register_forward_hook(_patch_hook(value)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def _capture_hook(name, cache, detach, retain_grad, capture_input):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        if retain_grad and tensor.requires_grad:
            tensor.retain_grad()
        cache[name] = tensor.detach().clone() if detach else tensor
        if capture_input and inputs:
            in_tensor = inputs[0]
            if isinstance(in_tensor, torch.Tensor):
                cache[f"{name}::input"] = in_tensor.detach().clone() if detach else in_tensor

    return hook


def _patch_hook(value):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (value, *output[1:])
        return value

    return hook
