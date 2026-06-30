"""LayerRegistry — discover and look up named submodules of a model."""
from __future__ import annotations

import re
from collections.abc import Callable

import torch.nn as nn


class LayerRegistry:
    """Indexes every named submodule of a model so analysis planes can refer
    to layers by string name (e.g. ``"transformer.h.4.mlp"``) instead of
    walking the module tree themselves.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._modules: dict[str, nn.Module] = {
            name: module for name, module in model.named_modules() if name
        }

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __getitem__(self, name: str) -> nn.Module:
        try:
            return self._modules[name]
        except KeyError as exc:
            raise KeyError(
                f"No layer named {name!r}. Use registry.names() to list "
                f"available layers, or registry.find(pattern) to search by regex."
            ) from exc

    def get(self, name: str) -> nn.Module | None:
        return self._modules.get(name)

    def names(self) -> list[str]:
        return list(self._modules.keys())

    def find(self, pattern: str) -> list[str]:
        """Return layer names matching a regex pattern."""
        rx = re.compile(pattern)
        return [name for name in self._modules if rx.search(name)]

    def of_type(self, module_type: type) -> list[str]:
        """Return layer names whose module is an instance of ``module_type``."""
        return [name for name, mod in self._modules.items() if isinstance(mod, module_type)]

    def filter(self, predicate: Callable[[str, nn.Module], bool]) -> list[str]:
        return [name for name, mod in self._modules.items() if predicate(name, mod)]

    def summary(self) -> list[dict]:
        """Per-layer type + parameter count, used by ``Inspector.summary()``."""
        rows = []
        for name, mod in self._modules.items():
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append({"name": name, "type": type(mod).__name__, "params": n_params})
        return rows
