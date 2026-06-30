"""Shared pytest fixtures: tiny, fast, network-free models for every plane."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / no-display environments

import pytest
import torch
import torch.nn as nn

from nndbg.inspector import Inspector


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def tiny_mlp() -> TinyMLP:
    torch.manual_seed(0)
    return TinyMLP()


@pytest.fixture
def mlp_inspector(tiny_mlp: TinyMLP) -> Inspector:
    return Inspector(tiny_mlp)


@pytest.fixture
def mlp_probe_dataset() -> list[tuple[torch.Tensor, int]]:
    torch.manual_seed(1)
    dataset = []
    for i in range(30):
        label = i % 2
        x = torch.randn(1, 4) + (label * 5)
        dataset.append((x, label))
    return dataset


@pytest.fixture
def tiny_transformer():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=100, n_positions=32, n_embd=16, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture
def transformer_inspector(tiny_transformer) -> Inspector:
    return Inspector(tiny_transformer)


@pytest.fixture
def token_dataset() -> list[torch.Tensor]:
    torch.manual_seed(2)
    return [torch.randint(0, 100, (1, 8)) for _ in range(24)]
