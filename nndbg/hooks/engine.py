from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nndbg.utils import get_logger

logger = get_logger(__name__)

class HookEngine:
    """
    Attaches and manages forward hooks on a PyTorch model.
    Captures activations from every named layer during a forward pass.

    Usage:
        engine = HookEngine(model)
        engine.attach()

        with engine.capture() as ctx:
            model(**inputs)

        activations = ctx.activations  # Dict[layer_name -> Tensor]
        engine.detach()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured: Dict[str, torch.Tensor] = {}
        self._layer_registry: Dict[str, nn.Module] = {}
        self._capturing = False

        self._discover_layers()


    def _discover_layers(self) -> None:
        self._layer_registry.clear()
        for name, module in self.model.named_modules():
            if name:
                self._layer_registry[name] = module
        logger.info(f"Discovered {len(self._layer_registry)} layers")


    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_registry.keys())
    

    def attach(
            self,
            layer_filter: Optional[Callable[[str,nn.Module], bool]] = None,
    ) -> None:
        self.detach()

        for name, module in self._layer_registry.items():
            if layer_filter is not None and not layer_filter(name, module):
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

        logger.info(f"Attached {len(self._hooks)} hooks")


    def _make_hook(self, layer_name: str) -> Callable:

        def hook(module: nn.Module, input: Tuple, output) -> None:
            if not self._capturing:
                return
            
            # Some layers output tuple so take out tensor first
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            if isinstance(tensor, torch.Tensor):
                self._captured[layer_name] = tensor.detach().cpu()

        return hook


    def detach(self) -> None:
        """Removes all attached hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("All hooks detached")

    
    def capture(self) -> "_CaptureContext":
        """
        Context manager for a single forward pass capture.

        Usage:
            with engine.capture() as ctx:
                model(**inputs)
            ctx.activations  # populated after exit
        """
        return _CaptureContext(self)        

    def _start_capture(self) -> None:
        self._captured.clear()
        self._capturing = True

    def _end_capture(self) -> Dict[str, torch.Tensor]:
        self._capturing = False
        return dict(self._captured)
    
class _CaptureContext:

    def __init__(self, engine: HookEngine):
        self._engine = engine
        self.activations: Dict[str, torch.Tensor] = {}

    def __enter__(self) -> "_CaptureContext":
        self._engine._start_capture()
        return self
    
    def __exit__(self, *args) -> None:
        self.activations = self._engine._end_capture()