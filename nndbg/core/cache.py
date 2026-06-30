"""ActivationCache — dict-like container for captured activations."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch


class ActivationCache:
    """Maps layer name -> activation tensor, with conveniences for saving
    and reloading a captured run to/from disk.
    """

    def __init__(self) -> None:
        self._data: dict[str, torch.Tensor] = {}

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def layers(self) -> list[str]:
        """Layer names present, excluding synthetic ``::input`` keys."""
        return [k for k in self._data if "::" not in k]

    def save(self, path: str | Path) -> None:
        torch.save(self._data, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        cache = cls()
        cache._data = torch.load(Path(path), map_location="cpu")
        return cache

    def __repr__(self) -> str:
        return f"ActivationCache({list(self._data.keys())})"
