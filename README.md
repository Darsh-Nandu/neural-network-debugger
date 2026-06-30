# NNDbg — Mechanistic Interpretability Toolkit

[![CI](https://github.com/Darsh-Nandu/neural-network-debugger/actions/workflows/ci.yml/badge.svg)](https://github.com/Darsh-Nandu/neural-network-debugger/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NNDbg is a researcher-first interpretability toolkit for PyTorch and HuggingFace models.
It answers both *where* concepts are encoded (probing) and *why* — which inputs caused a
behaviour (attribution, causal tracing) — and goes further into feature-level decomposition
via sparse autoencoders.

## Feature overview

| Plane | What it answers | Key method |
|-------|-----------------|------------|
| **Probing** | Where is concept X encoded? | `inspector.probing.fit(dataset, concept="sentiment")` |
| **Attribution** | Which tokens caused layer L to fire? | `inspector.attribution.saliency/ig/gradcam(...)` |
| **Attention** | What does each head attend to? | `inspector.attention.per_head_heatmap(input_ids)` |
| **Causal tracing** | Which layers *causally* produce a behaviour? | `inspector.patching.causal_trace(clean, corrupted)` |
| **Concept erasure** | Can I remove concept X from the representations? | `inspector.erasure.leace(dataset, concept="gender")` |
| **Geometry** | How does fine-tuning change representation structure? | `inspector.geometry.cka(dataset, layers_a=..., inspector_b=ft_model)` |
| **Neurons** | Which inputs maximally activate neuron N? | `inspector.neurons.top_activating_inputs(dataset, neuron_idx=42)` |
| **SAE** | What monosemantic features does this MLP learn? | `inspector.sae.train(dataset, layer=...)` |

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nndbg import Inspector

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inspector = Inspector(model, tokenizer)

# See what's available
inspector.summary()
inspector.layers()[:5]

# Probe for a concept
dataset = [(tokenizer(t, return_tensors="pt")["input_ids"], label)
           for t, label in [("I love this", 1), ("I hate this", 0)]]
report = inspector.probing.fit(dataset, concept="sentiment")
report.figure().show()

# Causal tracing
clean     = tokenizer("Paris is the capital of France", return_tensors="pt")["input_ids"]
corrupted = tokenizer("Berlin is the capital of France", return_tensors="pt")["input_ids"]
patch     = inspector.patching.causal_trace(clean, corrupted)
patch.figure().show()
```

## Installation

```bash
# Core
pip install nndbg

# With gradient attribution (captum)
pip install nndbg[attribution]

# With UMAP geometry
pip install nndbg[geometry]

# Everything
pip install nndbg[all]
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale and data flow.

## Contributing

```bash
git clone https://github.com/Darsh-Nandu/neural-network-debugger
cd neural-network-debugger
pip install -e ".[dev]"
pre-commit install
pytest tests/unit/
```