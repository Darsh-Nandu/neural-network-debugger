from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn

@dataclass
class LayerInfo:
    name: str
    layer_type: str
    param_count: int
    trainable: int
    frozen: int
    depth: int

class HookRegistry:

    def __init__(self, model: nn.Module):
        self._registry: Dict[str, LayerInfo] = {}
        self._build(model)

    def _build(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if not name:
                continue # Skip root module
            
            depth = name.count(".")
            param_count = sum(
                p.numel() for p in module.parameters(recurse=False)
            )
            trainable = sum(
                p.numel()
                for p in module.parameters(recurse=False)
                if p.requires_grad
            )
            self._registry[name] = LayerInfo(
                name=name,
                layer_type=type(module).__name__,
                param_count=param_count,
                trainable=trainable,
                frozen=param_count-trainable,
                depth=depth
            )
        
    def get(self, name: str) -> Optional[LayerInfo]:
        return self._registry.get(name)
    
    def all(self) -> List[LayerInfo]:
        return list(self._registry.values())
    
    def filter_by_type(self, *types: str) -> List[LayerInfo]:
        return [l for l in self._registry.values() if l.layer_type in types]
    
    def filter_by_depth(self, max_depth: int) -> List[LayerInfo]:
        return [l for l in self._registry.values() if l.depth <= max_depth]
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def __repr__(self) -> str:
        return f"HookRegistry({len(self._registry)} layers)"