"""
ModelProbe - the main entry point for NNDbg.

Usage:
    probe = ModelProbe.from_pretrained("google/mt5-small")

    probe.add_axis("language", {
        "english": ["The cat sat on the mat.", ...],
        "french":  ["Le chat était assis.", ...],
    })

    results = probe.run()
    results.show()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nndbg.hooks import HookEngine, HookRegistry
from nndbg.probing import Axis, ProbeTrainer
from nndbg.storage import ActivationStore
from nndbg.utils import get_device, get_logger

logger = get_logger(__name__)

class ModelProbe:
    """
    Attach to any PyTorch or HuggingFace model.
    Add comparison axes. Run deep activation analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        model_name: str = "unknown",
        store_path: str = ":memory:",
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length

        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

        self._hook_engine = HookEngine(model)
        self._registry = HookRegistry(model)
        self._store = ActivationStore(store_path)
        self._axes: List[Axis] = []

        logger.info(
            f"ModelProbe ready | model={model_name} | device={self.device}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        store_path: str = ":memory:",
        **kwargs,
    ) -> "ModelProbe":
        """Load any HuggingFace model by name."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        logger.info(f"Loading '{model_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        return cls(
            model,
            tokenizer,
            model_name=model_name,
            store_path=store_path,
            **kwargs,
        )
    
    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer = None,
        model_name: str = "custom",\
        **kwargs
    ) -> "ModelProbe":
        """Wrap an already loaded PyTorch model"""
        return cls(model, tokenizer, model_name=model_name, **kwargs)
    
    def add_axis(
        self,
        name: str,
        groups: Dict[str, List[str]],
    ) -> "ModelProbe":
        """
        Add a comparison axis.

        Args:
            name:   e.g. "language", "domain", "sentiment"
            groups: {"group_label": ["sample1", "sample2", ...]}

        Returns:
            self — so calls can be chained
        """
        axis = Axis(name=name, groups=groups)
        self._axes.append(axis)
        logger.info(f"Added axis: {axis}")
        return self
    
    def run(self):
        """
        Run the full analysis pipeline:

        1. Attach hooks to all layers
        2. For each axis / group / sample → run forward pass
        3. Store activation stats in DuckDB
        4. Train linear probes per layer per axis
        5. Return ProbeResults

        Returns:
            ProbeResults
        """
        if not self._axes:
            raise ValueError(
                "No axes added. Use probe.add_axis() before calling run()."
            )
        
        from nndbg.results import ProbeResults

        run_id = self._store.create_run(
            self.model_name,
            {
                "axes": [ax.name for ax in self._axes],
                "max_length": self.max_length
            }
        )

        self._hook_engine.attach()

    def _run_single():
        ...
