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
from nndbg.probing import Axis
from nndbg.storage import ActivationStore
from nndbg.probing.trainer import ProbeTrainer as _ProbeTrainer 
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
        probe_trainer: Optional["ProbeTrainer"] = None,
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
        self._probe_trainer = probe_trainer if probe_trainer is not None else _ProbeTrainer()
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

        # { axis_name -> { layer_name -> { group_name -> [stats_dict] } } }
        layer_group_data: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}

        # Step 1: collect activations
        for axis in self._axes:
            logger.info(f"Processing axis: '{axis.name}'")
            layer_group_data[axis.name] = {}

            for group_name, texts in axis.groups.items():
                logger.info(f"Group '{group_name}': {len(texts)} samples")

                for sample_idx, text in enumerate(tqdm(texts, desc=f"{axis.name}/{group_name}")):
                    activations = self._run_single(text)

                    for layer_name, tensor in activations.items():

                        # Store to DB
                        self._store.store_activation(
                            run_id=run_id,
                            axis_name=axis.name,
                            group_name=group_name,
                            sample_idx=sample_idx,
                            layer_name=layer_name,
                            tensor=tensor
                        )

                        # Build in memory structure for probe trainer
                        axis_data = layer_group_data[axis.name]
                        if layer_name not in axis_data:
                            axis_data[layer_name] = {}
                        if group_name not in axis_data[layer_name]:
                            axis_data[layer_name][group_name] = []

                        arr = tensor.float().numpy().flatten()
                        arr = np.nan_to_num(arr)
                        axis_data[layer_name][group_name].append({
                            "mean":     float(arr.mean()),
                            "std":      float(arr.std()),
                            "l2_norm":  float(np.linalg.norm(arr)),
                            "sparsity": float((arr == 0).mean()),
                            "min_val":  float(arr.min()),
                            "max_val":  float(arr.max()),
                        })

        self._hook_engine.detach()

        # Step 2: train probe
        logger.info("Training linear probes ...")
        trainer = self._probe_trainer
        probe_scores: Dict[str, Dict[str, float]] = {}

        for axis in self._axes:
            scores = trainer.train_all_layers(layer_group_data[axis.name])
            probe_scores[axis.name] = scores

            for layer_name, score in scores.items():
                self._store.store_layer_stat(
                    run_id=run_id,
                    axis_name=axis.name,
                    layer_name=layer_name,
                    group_name="all",
                    probe_score=score,
                    mean_diff=0.0
                )
        logger.info("Analysis complete.")

        return ProbeResults(
            run_id=run_id,
            model_name=self.model_name,
            axes=self._axes,
            probe_scores=probe_scores,
            store=self._store,
            layer_group_data=layer_group_data
        )
    def _run_single(self, text: str) -> Dict[str, torch.Tensor]:
        """Run one text sample through the model, return activations."""
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer found. Pass tokenizer= to ModelProbe() "
                "or use from_pretrained()."
            )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            with self._hook_engine.capture() as ctx:
                try:
                    self.model(**inputs)
                except Exception as e:
                    logger.warning(f"Forward pass issue: {e}")

        return ctx.activations
    
    def architecture(self) -> Dict:
        return self._hook_engine.architecture_summary()
    
    def __repr__(self) -> str:
        axes = ", ".join(f"'{a.name}'" for a in self._axes)
        return f"ModelProbe(model='{self.model_name}', axes=[{axes}])"